//! Pole detection and corridors (W3): what a pole says about the wires.
//!
//! A conductor is thin, and fades at about 100 m. The pole holding it fills
//! the beam, so it is seen as far as a surface is. Three things follow:
//!
//! - **A pole is a stack of returns.** Points are kept by ground cell. A cell
//!   whose points stand [`PoleConfig::min_height_m`] tall with few gaps is a
//!   pole. A wire crossing a cell stays at one height, so stacked conductors
//!   leave gaps, and two lines crossing leave more. Nothing says which object
//!   a point came from.
//! - **A pole ends a wire track.** The corner in a wire at its support is what
//!   let a track fit through an unseen pole (W2). A detected pole splits such
//!   a track, and stops the pieces joining again
//!   ([`crate::wires::WireTracker::set_poles`]).
//! - **Poles make corridors.** A track between two poles, or a straight row of
//!   them, says where wires run, including the stretches no return came from.
//!   The last pole in a row says where the next span goes, and the free end of
//!   a track says the wire goes on. Those are predictions, and are given the
//!   clearance of something not measured.
//!
//! A pole's top is the highest return. A LiDAR looking below the horizon does
//! not see above itself, so a pole never seen from above its top reports
//! [`Pole::top_open`]: the top is at least that high.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::wires::WireEstimate;

/// Returns one scan must put on a pole to show where its top is.
const MIN_TOP_POINTS: u32 = 8;

/// What the pole detector and corridor builder are declared to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoleConfig {
    /// A column of returns is this wide across the ground, metres.
    pub cell_m: f64,
    /// A column's height bins, metres.
    pub height_bin_m: f64,
    /// Points a column needs to be a pole.
    pub min_points: u32,
    /// How tall a column's returns must stand to be a pole, metres.
    pub min_height_m: f64,
    /// The share of a column's height bins that must hold a return.
    pub fill: f64,
    /// Columns this close together are one pole, metres: a pole on the
    /// boundary of two cells fills both.
    pub merge_radius_m: f64,
    /// A return this close to a known pole is the pole's, metres.
    pub claim_radius_m: f64,
    /// Columns kept. The least-filled go first.
    pub column_capacity: usize,
    /// Smallest one-sigma position error a pole claims, metres.
    pub floor_sigma_m: f64,
    /// A top within this of the highest observer is unproven, metres.
    pub top_margin_m: f64,
    /// A wire track ends at a pole within this across its line, metres. It
    /// has to hold the widest cross-arm.
    pub lateral_gate_m: f64,
    /// A wire track looks for its pole this far past its evidence, metres.
    pub reach_m: f64,
    /// Longest span two poles can be linked across, metres.
    pub max_span_m: f64,
    /// Shortest span two poles can be linked across, metres.
    pub min_span_m: f64,
    /// A row of poles bends by at most this at a pole, radians.
    pub row_angle_rad: f64,
    /// The sag assumed where only poles are known, as a share of the span.
    pub assumed_sag_ratio: f64,
    /// One-sigma error of a stretch that is predicted, not measured, metres.
    pub predicted_sigma_m: f64,
    /// How far a free track end is carried on, metres.
    pub stub_m: f64,
    /// How far above a free end the wire may climb to its support, metres.
    pub stub_rise_m: f64,
    /// How close to a pole's axis a platform is kept, metres.
    pub keep_out_m: f64,
    /// How far above its highest return an open top is assumed, metres. A
    /// pole seen only from below it can be much taller than it looks.
    pub open_top_m: f64,
}

impl Default for PoleConfig {
    fn default() -> Self {
        Self {
            cell_m: 4.0,
            height_bin_m: 2.0,
            min_points: 12,
            min_height_m: 8.0,
            fill: 0.5,
            merge_radius_m: 6.0,
            claim_radius_m: 1.5,
            column_capacity: 8192,
            floor_sigma_m: 0.25,
            top_margin_m: 3.0,
            lateral_gate_m: 8.0,
            reach_m: 120.0,
            max_span_m: 600.0,
            min_span_m: 30.0,
            row_angle_rad: 0.2,
            assumed_sag_ratio: 0.03,
            predicted_sigma_m: 3.0,
            stub_m: 120.0,
            stub_rise_m: 3.0,
            keep_out_m: 6.0,
            open_top_m: 30.0,
        }
    }
}

impl PoleConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("cell_m", self.cell_m),
            ("height_bin_m", self.height_bin_m),
            ("min_height_m", self.min_height_m),
            ("merge_radius_m", self.merge_radius_m),
            ("claim_radius_m", self.claim_radius_m),
            ("floor_sigma_m", self.floor_sigma_m),
            ("lateral_gate_m", self.lateral_gate_m),
            ("max_span_m", self.max_span_m),
            ("min_span_m", self.min_span_m),
            ("row_angle_rad", self.row_angle_rad),
            ("predicted_sigma_m", self.predicted_sigma_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("pole {name} must be positive, got {value}"));
            }
        }
        for (name, value) in [
            ("top_margin_m", self.top_margin_m),
            ("reach_m", self.reach_m),
            ("assumed_sag_ratio", self.assumed_sag_ratio),
            ("stub_m", self.stub_m),
            ("stub_rise_m", self.stub_rise_m),
            ("keep_out_m", self.keep_out_m),
            ("open_top_m", self.open_top_m),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("pole {name} must not be negative, got {value}"));
            }
        }
        if !(self.fill > 0.0 && self.fill <= 1.0) {
            return Err(format!("pole fill must be in (0, 1], got {}", self.fill));
        }
        if self.min_points == 0 {
            return Err("pole min_points must be at least 1".into());
        }
        if self.column_capacity == 0 {
            return Err("pole column_capacity must be at least 1".into());
        }
        if self.min_span_m >= self.max_span_m {
            return Err("pole min_span_m must be below max_span_m".into());
        }
        Ok(())
    }
}

/// One ground cell's returns.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
struct Column {
    n: u32,
    sum_xy: [f64; 2],
    sum_sq: f64,
    low_z_m: f64,
    high_z_m: f64,
    /// Height bin of bit 0.
    anchor: i32,
    /// Bits: which of 128 bins from the anchor hold a return.
    bins: [u64; 2],
}

impl Column {
    fn new(point: [f64; 3], bin_m: f64) -> Self {
        Self {
            n: 0,
            sum_xy: [0.0; 2],
            sum_sq: 0.0,
            low_z_m: point[2],
            high_z_m: point[2],
            anchor: (point[2] / bin_m).floor() as i32 - 64,
            bins: [0; 2],
        }
    }

    fn add(&mut self, point: [f64; 3], bin_m: f64) {
        self.n += 1;
        self.sum_xy[0] += point[0];
        self.sum_xy[1] += point[1];
        self.sum_sq += point[0] * point[0] + point[1] * point[1];
        self.low_z_m = self.low_z_m.min(point[2]);
        self.high_z_m = self.high_z_m.max(point[2]);
        let bit = (point[2] / bin_m).floor() as i32 - self.anchor;
        if (0..128).contains(&bit) {
            self.bins[(bit / 64) as usize] |= 1 << (bit % 64);
        }
    }

    /// Whether these returns stand like a pole.
    fn stands(&self, config: &PoleConfig) -> bool {
        if self.n < config.min_points || self.high_z_m - self.low_z_m < config.min_height_m {
            return false;
        }
        let spanned = ((self.high_z_m - self.low_z_m) / config.height_bin_m).ceil();
        let filled = self.bins[0].count_ones() + self.bins[1].count_ones();
        f64::from(filled) >= config.fill * spanned
    }
}

/// A detected pole.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pole {
    pub id: u32,
    pub n: u32,
    sum_xy: [f64; 2],
    sum_sq: f64,
    /// The lowest return, metres: the pole's foot, or as far down as it was
    /// seen.
    pub low_z_m: f64,
    /// The highest return, metres.
    pub high_z_m: f64,
    /// The highest return a scan from above showed as the top, metres: a
    /// scan of a few points reaching it, from a sensor clear above it.
    #[serde(default)]
    proven_top_z_m: Option<f64>,
    /// True when the top has not been seen from above.
    #[serde(default)]
    pub top_open: bool,
}

impl Pole {
    pub fn xy_m(&self) -> [f64; 2] {
        let n = f64::from(self.n);
        [self.sum_xy[0] / n, self.sum_xy[1] / n]
    }

    /// One-sigma error of the position, metres.
    pub fn sigma_m(&self, floor_m: f64) -> f64 {
        let n = f64::from(self.n);
        let [x, y] = self.xy_m();
        let variance = (self.sum_sq / n - x * x - y * y).max(0.0);
        (variance / n).sqrt().max(floor_m)
    }

    /// The height to plan for: the top, or above it while it is unproven.
    pub fn planning_top_m(&self, config: &PoleConfig) -> f64 {
        self.high_z_m
            + if self.top_open {
                config.open_top_m
            } else {
                0.0
            }
    }

    fn absorb(&mut self, n: u32, sum_xy: [f64; 2], sum_sq: f64, low_z_m: f64, high_z_m: f64) {
        self.n += n;
        self.sum_xy[0] += sum_xy[0];
        self.sum_xy[1] += sum_xy[1];
        self.sum_sq += sum_sq;
        self.low_z_m = self.low_z_m.min(low_z_m);
        self.high_z_m = self.high_z_m.max(high_z_m);
    }
}

/// What a wire track needs to know of a pole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoleAnchor {
    pub id: u32,
    pub xy_m: [f64; 2],
}

/// Every pole seen so far.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PoleDetector {
    config: PoleConfig,
    /// By ground cell, x in the high half and y in the low: a checkpoint's
    /// map keys must be plain.
    columns: BTreeMap<i64, Column>,
    poles: Vec<Pole>,
    next_id: u32,
}

impl PoleDetector {
    pub fn new(config: PoleConfig) -> Self {
        Self {
            config,
            columns: BTreeMap::new(),
            poles: Vec::new(),
            next_id: 0,
        }
    }

    pub fn config(&self) -> &PoleConfig {
        &self.config
    }

    pub fn poles(&self) -> &[Pole] {
        &self.poles
    }

    pub fn anchors(&self) -> Vec<PoleAnchor> {
        self.poles
            .iter()
            .map(|pole| PoleAnchor {
                id: pole.id,
                xy_m: pole.xy_m(),
            })
            .collect()
    }

    /// Back to nothing known.
    pub fn clear(&mut self) {
        *self = Self::new(self.config.clone());
    }

    /// Take a scan's points from a sensor at `observer_z_m`, and give back
    /// those that are not on a known pole, for the wire tracker.
    pub fn add_points(&mut self, observer_z_m: f64, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mut rest = Vec::with_capacity(points.len());
        // Each pole's returns in this scan, and the highest.
        let mut seen: BTreeMap<usize, (u32, f64)> = BTreeMap::new();
        for &point in points {
            match self.claiming(point) {
                Some(k) => {
                    let [x, y, z] = point;
                    self.poles[k].absorb(1, [x, y], x * x + y * y, z, z);
                    let entry = seen.entry(k).or_insert((0, f64::NEG_INFINITY));
                    entry.0 += 1;
                    entry.1 = entry.1.max(z);
                }
                None => {
                    self.add_to_column(point);
                    rest.push(point);
                }
            }
        }
        let margin = self.config.top_margin_m;
        for (k, (count, high)) in seen {
            let pole = &mut self.poles[k];
            if count >= MIN_TOP_POINTS
                && high >= pole.high_z_m - margin
                && observer_z_m >= pole.high_z_m + margin
            {
                pole.proven_top_z_m = Some(pole.high_z_m);
            }
        }
        for pole in &mut self.poles {
            pole.top_open = pole
                .proven_top_z_m
                .is_none_or(|top| pole.high_z_m > top + margin);
        }
        rest
    }

    fn claiming(&self, point: [f64; 3]) -> Option<usize> {
        let reach = self.config.claim_radius_m;
        self.poles.iter().position(|pole| {
            let [x, y] = pole.xy_m();
            (point[0] - x).hypot(point[1] - y) <= reach
        })
    }

    fn add_to_column(&mut self, point: [f64; 3]) {
        let cell = |v: f64| (v / self.config.cell_m).floor() as i32;
        let key = (i64::from(cell(point[0])) << 32) | i64::from(cell(point[1]) as u32);
        let bin_m = self.config.height_bin_m;
        let column = self
            .columns
            .entry(key)
            .or_insert_with(|| Column::new(point, bin_m));
        column.add(point, bin_m);
        if column.stands(&self.config) {
            let column = self.columns.remove(&key).expect("just filled");
            self.promote(column);
        } else if self.columns.len() > self.config.column_capacity {
            let least = self
                .columns
                .iter()
                .min_by_key(|(_, column)| column.n)
                .map(|(key, _)| *key)
                .expect("over capacity");
            self.columns.remove(&least);
        }
    }

    /// A column that stands is a pole, or part of one beside it.
    fn promote(&mut self, column: Column) {
        let n = f64::from(column.n);
        let centre = [column.sum_xy[0] / n, column.sum_xy[1] / n];
        let reach = self.config.merge_radius_m;
        let beside = self.poles.iter_mut().find(|pole| {
            let [x, y] = pole.xy_m();
            (centre[0] - x).hypot(centre[1] - y) <= reach
        });
        match beside {
            Some(pole) => {
                pole.absorb(
                    column.n,
                    column.sum_xy,
                    column.sum_sq,
                    column.low_z_m,
                    column.high_z_m,
                );
            }
            None => {
                self.poles.push(Pole {
                    id: self.next_id,
                    n: column.n,
                    sum_xy: column.sum_xy,
                    sum_sq: column.sum_sq,
                    low_z_m: column.low_z_m,
                    high_z_m: column.high_z_m,
                    proven_top_z_m: None,
                    top_open: true,
                });
                self.next_id += 1;
            }
        }
    }
}

/// What a corridor span rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorridorSource {
    /// Tracked wire runs between two poles.
    Tracks,
    /// Two poles of a straight row, whether or not wire was seen.
    Row,
    /// The span past the last pole of a row: not seen.
    PredictedSpan,
    /// A wire past the end of its evidence: not seen.
    PredictedStub,
}

/// One stretch of ground that wires believed to be there run along.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorridorSpan {
    /// A stable name: the same stretch keeps it as evidence lands.
    pub key: String,
    pub source: CorridorSource,
    pub from_pole: Option<u32>,
    pub to_pole: Option<u32>,
    /// The wires' height at each end, and where it is on the ground.
    pub from_m: [f64; 3],
    pub to_m: [f64; 3],
    /// The widest wire off the centre line, metres.
    pub half_width_m: f64,
    /// One-sigma error, metres.
    pub sigma_m: f64,
    /// How far the wires hang below the chord, as a share of the span.
    pub sag_ratio: f64,
}

fn distance_xy(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn unit(from: [f64; 2], to: [f64; 2]) -> Option<[f64; 2]> {
    let length = distance_xy(from, to);
    (length > 1e-6).then(|| [(to[0] - from[0]) / length, (to[1] - from[1]) / length])
}

/// How far `point` is off the line through `from` along `direction`, and how
/// far along it.
fn off_line(point: [f64; 2], from: [f64; 2], direction: [f64; 2]) -> (f64, f64) {
    let (dx, dy) = (point[0] - from[0], point[1] - from[1]);
    (
        dx * direction[0] + dy * direction[1],
        (dy * direction[0] - dx * direction[1]).abs(),
    )
}

/// The corridors the poles and tracks imply, in a stable order. Derived: it
/// is built again from what is known, so a stretch that stops being implied
/// goes.
pub fn corridors(
    poles: &[Pole],
    estimates: &[WireEstimate],
    config: &PoleConfig,
) -> Vec<CorridorSpan> {
    let by_id: BTreeMap<u32, &Pole> = poles.iter().map(|pole| (pole.id, pole)).collect();
    let confirmed = || estimates.iter().filter(|estimate| estimate.confirmed);
    // Linked pairs, low id first, with the widest wire off the centre line.
    let mut links: BTreeMap<(u32, u32), (CorridorSource, f64)> = BTreeMap::new();
    for track in confirmed() {
        let [Some(a), Some(b)] = track.pole_ends else {
            continue;
        };
        if a == b {
            continue;
        }
        let (Some(pa), Some(pb)) = (by_id.get(&a), by_id.get(&b)) else {
            continue;
        };
        let Some(direction) = unit(pa.xy_m(), pb.xy_m()) else {
            continue;
        };
        let (_, off_from) = off_line([track.from_m[0], track.from_m[1]], pa.xy_m(), direction);
        let (_, off_to) = off_line([track.to_m[0], track.to_m[1]], pa.xy_m(), direction);
        let entry = links
            .entry((a.min(b), a.max(b)))
            .or_insert((CorridorSource::Tracks, 0.0));
        entry.1 = entry.1.max(off_from).max(off_to);
    }
    // A track with one pole, whose line runs on to another.
    for track in confirmed() {
        let (pole_id, from, to) = match track.pole_ends {
            [Some(a), None] => (a, track.from_m, track.to_m),
            [None, Some(b)] => (b, track.to_m, track.from_m),
            _ => continue,
        };
        let Some(direction) = unit([from[0], from[1]], [to[0], to[1]]) else {
            continue;
        };
        let Some(start) = by_id.get(&pole_id) else {
            continue;
        };
        let next = poles
            .iter()
            .filter(|pole| pole.id != pole_id)
            .filter_map(|pole| {
                let (along, off) = off_line(pole.xy_m(), start.xy_m(), direction);
                (off <= config.lateral_gate_m
                    && along >= config.min_span_m
                    && along <= config.max_span_m)
                    .then_some((along, pole.id))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if let Some((_, other)) = next {
            links
                .entry((pole_id.min(other), pole_id.max(other)))
                .or_insert((CorridorSource::Tracks, 0.0));
        }
    }
    // Straight rows: each pole with a neighbour on either side of it.
    for middle in poles {
        let at = middle.xy_m();
        let near: Vec<&Pole> = poles
            .iter()
            .filter(|pole| pole.id != middle.id)
            .filter(|pole| {
                let d = distance_xy(pole.xy_m(), at);
                d >= config.min_span_m && d <= config.max_span_m
            })
            .collect();
        for (k, a) in near.iter().enumerate() {
            for b in &near[k + 1..] {
                let (Some(ua), Some(ub)) = (unit(at, a.xy_m()), unit(at, b.xy_m())) else {
                    continue;
                };
                let cos = (ua[0] * ub[0] + ua[1] * ub[1]).clamp(-1.0, 1.0);
                if std::f64::consts::PI - cos.acos() > config.row_angle_rad {
                    continue;
                }
                let (da, db) = (distance_xy(at, a.xy_m()), distance_xy(at, b.xy_m()));
                if da.max(db) > 2.0 * da.min(db) {
                    continue;
                }
                for other in [a, b] {
                    links
                        .entry((middle.id.min(other.id), middle.id.max(other.id)))
                        .or_insert((CorridorSource::Row, 0.0));
                }
            }
        }
    }
    // A row's link can step over a pole: keep only neighbours.
    let pairs: Vec<(u32, u32)> = links.keys().copied().collect();
    for (a, b) in pairs {
        let (pa, pb) = (by_id[&a].xy_m(), by_id[&b].xy_m());
        let Some(direction) = unit(pa, pb) else {
            continue;
        };
        let length = distance_xy(pa, pb);
        let stepped = links
            .get(&(a, b))
            .is_some_and(|link| link.0 == CorridorSource::Row)
            && poles.iter().any(|pole| {
                let (along, off) = off_line(pole.xy_m(), pa, direction);
                pole.id != a
                    && pole.id != b
                    && off <= config.lateral_gate_m
                    && along > 1.0
                    && along < length - 1.0
            });
        if stepped {
            links.remove(&(a, b));
        }
    }
    // A narrow sensor clips a pole's top unless it passes close, so one pole
    // can look far shorter than it is. Wires on a line hang at much the same
    // height, so a connected row plans for the highest evidence in it:
    // any pole's top, or any track on it.
    let mut group: BTreeMap<u32, u32> = poles.iter().map(|pole| (pole.id, pole.id)).collect();
    fn root(group: &BTreeMap<u32, u32>, mut id: u32) -> u32 {
        while group[&id] != id {
            id = group[&id];
        }
        id
    }
    for &(a, b) in links.keys() {
        let (ra, rb) = (root(&group, a), root(&group, b));
        group.insert(ra.max(rb), ra.min(rb));
    }
    let mut row_top: BTreeMap<u32, f64> = BTreeMap::new();
    for pole in poles {
        let top = row_top
            .entry(root(&group, pole.id))
            .or_insert(f64::NEG_INFINITY);
        *top = top.max(pole.planning_top_m(config));
    }
    for track in confirmed() {
        for id in track.pole_ends.iter().flatten() {
            if group.contains_key(id) {
                let height = track.from_m[2].max(track.to_m[2]);
                let top = row_top.entry(root(&group, *id)).or_insert(height);
                *top = top.max(height);
            }
        }
    }
    let top_of = |id: u32| row_top[&root(&group, id)];
    let sigma_of = |pole: &Pole| pole.sigma_m(config.floor_sigma_m);
    let mut spans = Vec::new();
    let mut degree: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&(a, b), &(source, width)) in &links {
        let (pa, pb) = (by_id[&a], by_id[&b]);
        let ([ax, ay], [bx, by]) = (pa.xy_m(), pb.xy_m());
        let top = top_of(a);
        degree.entry(a).or_default().push(b);
        degree.entry(b).or_default().push(a);
        spans.push(CorridorSpan {
            key: format!("{a}-{b}"),
            source,
            from_pole: Some(a),
            to_pole: Some(b),
            from_m: [ax, ay, top],
            to_m: [bx, by, top],
            half_width_m: width,
            sigma_m: sigma_of(pa).max(sigma_of(pb)),
            sag_ratio: config.assumed_sag_ratio,
        });
    }
    // The span past the last pole of a row, as long as the one before it.
    for (&end, others) in &degree {
        let &[previous] = others.as_slice() else {
            continue;
        };
        let (last, before) = (by_id[&end], by_id[&previous]);
        let Some(direction) = unit(before.xy_m(), last.xy_m()) else {
            continue;
        };
        let length = distance_xy(before.xy_m(), last.xy_m());
        let [x, y] = last.xy_m();
        spans.push(CorridorSpan {
            key: format!("{end}-from-{previous}"),
            source: CorridorSource::PredictedSpan,
            from_pole: Some(end),
            to_pole: None,
            from_m: [x, y, top_of(end)],
            to_m: [
                x + direction[0] * length,
                y + direction[1] * length,
                top_of(end),
            ],
            half_width_m: 0.0,
            sigma_m: config.predicted_sigma_m,
            sag_ratio: config.assumed_sag_ratio,
        });
    }
    // Where a wire's evidence stops and no pole does, the wire goes on.
    for track in confirmed() {
        for (end, reached, inward) in [(0, track.from_m, track.to_m), (1, track.to_m, track.from_m)]
        {
            if track.pole_ends[end].is_some() {
                continue;
            }
            let Some(direction) = unit([inward[0], inward[1]], [reached[0], reached[1]]) else {
                continue;
            };
            let z = reached[2] + config.stub_rise_m;
            spans.push(CorridorSpan {
                key: format!("t{}-{end}", track.id),
                source: CorridorSource::PredictedStub,
                from_pole: None,
                to_pole: None,
                from_m: [reached[0], reached[1], z],
                to_m: [
                    reached[0] + direction[0] * config.stub_m,
                    reached[1] + direction[1] * config.stub_m,
                    z,
                ],
                half_width_m: 0.0,
                sigma_m: config.predicted_sigma_m,
                sag_ratio: 0.0,
            });
        }
    }
    spans.sort_by(|a, b| a.key.cmp(&b.key));
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(detector: &mut PoleDetector, xy: [f64; 2], low: f64, high: f64, n: usize) {
        let mut points = Vec::new();
        for k in 0..n {
            let t = k as f64 / (n - 1) as f64;
            let wobble = 0.1 * ((k as f64) * 1.7).sin();
            points.push([xy[0] + wobble, xy[1] - wobble, low + (high - low) * t]);
        }
        detector.add_points(high + 20.0, &points);
    }

    fn detector() -> PoleDetector {
        PoleDetector::new(PoleConfig::default())
    }

    fn pole_at(id: u32, x: f64, y: f64, top: f64) -> Pole {
        Pole {
            id,
            n: 100,
            sum_xy: [x * 100.0, y * 100.0],
            sum_sq: (x * x + y * y) * 100.0,
            low_z_m: 0.0,
            high_z_m: top,
            proven_top_z_m: Some(top),
            top_open: false,
        }
    }

    fn track(id: u32, from: [f64; 3], to: [f64; 3], ends: [Option<u32>; 2]) -> WireEstimate {
        WireEstimate {
            id,
            from_m: from,
            to_m: to,
            sag_m: 1.0,
            sigma_lateral_m: 0.2,
            sigma_vertical_m: 0.2,
            observed_length_m: 100.0,
            points: 100,
            camera_support: 0,
            triangulated: false,
            confirmed: true,
            pole_ends: ends,
        }
    }

    #[test]
    fn a_stack_of_returns_is_a_pole() {
        let mut detector = detector();
        stack(&mut detector, [10.0, 20.0], 100.0, 130.0, 40);
        assert_eq!(detector.poles().len(), 1);
        let [x, y] = detector.poles()[0].xy_m();
        assert!((x - 10.0).abs() < 0.5 && (y - 20.0).abs() < 0.5);
    }

    /// Conductors at two heights in a cell stand tall enough but leave a gap
    /// a pole does not, and a level wire does not stand at all.
    #[test]
    fn wires_are_not_poles() {
        let mut level = detector();
        let points: Vec<[f64; 3]> = (0..200)
            .map(|k| [0.02 * f64::from(k), 1.0, 50.0 + 0.001 * f64::from(k)])
            .collect();
        level.add_points(80.0, &points);
        assert!(level.poles().is_empty());
        let mut stacked = detector();
        let points: Vec<[f64; 3]> = (0..200)
            .map(|k| [1.0, 1.0, if k % 2 == 0 { 40.0 } else { 60.0 }])
            .collect();
        stacked.add_points(80.0, &points);
        assert!(stacked.poles().is_empty());
    }

    #[test]
    fn a_pole_on_a_cell_boundary_is_one_pole() {
        let mut detector = detector();
        stack(&mut detector, [3.95, 2.0], 0.0, 30.0, 60);
        stack(&mut detector, [4.05, 2.0], 0.0, 30.0, 60);
        assert_eq!(detector.poles().len(), 1);
    }

    #[test]
    fn a_known_pole_keeps_its_points_from_the_tracker() {
        let mut detector = detector();
        stack(&mut detector, [0.5, 0.5], 0.0, 30.0, 40);
        let rest = detector.add_points(50.0, &[[0.6, 0.5, 12.0], [40.0, 0.0, 12.0]]);
        assert_eq!(rest, vec![[40.0, 0.0, 12.0]]);
    }

    /// A LiDAR looking below the horizon does not see above itself.
    #[test]
    fn a_top_seen_only_from_below_it_stays_open() {
        let mut detector = detector();
        let points: Vec<[f64; 3]> = (0..40)
            .map(|k| [0.5, 0.5, 10.0 + f64::from(k) / 2.0])
            .collect();
        detector.add_points(29.0, &points);
        assert!(detector.poles()[0].top_open);
        // From above, a few returns not reaching it prove nothing.
        detector.add_points(60.0, &[[0.5, 0.5, 15.0]; 12]);
        assert!(detector.poles()[0].top_open);
        let top: Vec<[f64; 3]> = (0..12)
            .map(|k| [0.5, 0.5, 28.0 + f64::from(k) / 12.0])
            .collect();
        detector.add_points(60.0, &top);
        assert!(!detector.poles()[0].top_open);
    }

    #[test]
    fn an_open_top_is_planned_for_above_what_was_seen() {
        let config = PoleConfig::default();
        let mut pole = pole_at(0, 0.0, 0.0, 30.0);
        assert_eq!(pole.planning_top_m(&config), 30.0);
        pole.top_open = true;
        assert_eq!(pole.planning_top_m(&config), 60.0);
    }

    #[test]
    fn the_columns_are_bounded() {
        let mut detector = PoleDetector::new(PoleConfig {
            column_capacity: 16,
            ..PoleConfig::default()
        });
        for k in 0..100 {
            detector.add_points(50.0, &[[5.0 * f64::from(k), 0.0, 10.0]]);
        }
        assert!(detector.columns.len() <= 17);
    }

    #[test]
    fn a_track_between_two_poles_links_them() {
        let config = PoleConfig::default();
        let poles = [pole_at(0, 0.0, 0.0, 30.0), pole_at(1, 200.0, 0.0, 30.0)];
        let tracks = [track(
            0,
            [0.0, 2.0, 30.0],
            [200.0, 2.0, 30.0],
            [Some(0), Some(1)],
        )];
        let spans = corridors(&poles, &tracks, &config);
        let span = spans
            .iter()
            .find(|s| s.source == CorridorSource::Tracks)
            .expect("linked");
        assert_eq!((span.from_pole, span.to_pole), (Some(0), Some(1)));
        assert!((span.half_width_m - 2.0).abs() < 1e-9);
    }

    /// A straight row is linked between neighbours with no wire seen, and
    /// the next span is predicted past each end.
    #[test]
    fn a_row_of_poles_implies_a_corridor_and_its_next_span() {
        let config = PoleConfig::default();
        let poles = [
            pole_at(0, 0.0, 0.0, 30.0),
            pole_at(1, 200.0, 2.0, 30.0),
            pole_at(2, 400.0, 0.0, 30.0),
        ];
        let spans = corridors(&poles, &[], &config);
        let rows: Vec<&CorridorSpan> = spans
            .iter()
            .filter(|s| s.source == CorridorSource::Row)
            .collect();
        assert_eq!(rows.len(), 2, "{spans:#?}");
        let predicted: Vec<&CorridorSpan> = spans
            .iter()
            .filter(|s| s.source == CorridorSource::PredictedSpan)
            .collect();
        assert_eq!(predicted.len(), 2);
        assert!(predicted.iter().any(|s| s.to_m[0] > 590.0));
        assert!(predicted.iter().any(|s| s.to_m[0] < -190.0));
        assert!(predicted[0].sigma_m > rows[0].sigma_m);
    }

    /// One pole seen only in part does not lower the wires of its row.
    #[test]
    fn a_row_plans_for_its_highest_evidence() {
        let config = PoleConfig::default();
        let poles = [
            pole_at(0, 0.0, 0.0, 190.0),
            pole_at(1, 200.0, 0.0, 260.0),
            pole_at(2, 400.0, 0.0, 190.0),
            pole_at(3, 600.0, 900.0, 190.0),
        ];
        let spans = corridors(&poles, &[], &config);
        assert!(!spans.is_empty());
        for span in spans {
            assert_eq!(span.from_m[2], 260.0, "{}", span.key);
            assert_eq!(span.to_m[2], 260.0, "{}", span.key);
        }
    }

    #[test]
    fn scattered_poles_imply_nothing() {
        let config = PoleConfig::default();
        let poles = [
            pole_at(0, 0.0, 0.0, 30.0),
            pole_at(1, 200.0, 0.0, 30.0),
            pole_at(2, 200.0, 200.0, 30.0),
        ];
        assert!(corridors(&poles, &[], &config).is_empty());
    }

    #[test]
    fn a_free_track_end_is_carried_on() {
        let config = PoleConfig::default();
        let poles = [pole_at(0, 0.0, 0.0, 30.0)];
        let tracks = [track(
            3,
            [0.0, 0.0, 30.0],
            [100.0, 0.0, 28.0],
            [Some(0), None],
        )];
        let spans = corridors(&poles, &tracks, &config);
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].source, CorridorSource::PredictedStub);
        assert!((spans[0].to_m[0] - 220.0).abs() < 1e-6);
        assert!((spans[0].to_m[2] - 31.0).abs() < 1e-6);
    }

    #[test]
    fn an_unconfirmed_track_implies_nothing() {
        let config = PoleConfig::default();
        let mut unconfirmed = track(1, [0.0, 0.0, 30.0], [100.0, 0.0, 30.0], [None, None]);
        unconfirmed.confirmed = false;
        assert!(corridors(&[], &[unconfirmed], &config).is_empty());
    }

    #[test]
    fn a_detector_survives_a_checkpoint() {
        let mut detector = detector();
        stack(&mut detector, [1.0, 1.0], 0.0, 30.0, 40);
        detector.add_points(50.0, &[[40.0, 0.0, 12.0]]);
        let text = serde_json::to_string(&detector).unwrap();
        let restored: PoleDetector = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, detector);
    }

    #[test]
    fn nonsense_is_refused() {
        let bad = PoleConfig {
            fill: 1.5,
            ..PoleConfig::default()
        };
        assert!(bad.validate().is_err());
        let bad = PoleConfig {
            min_span_m: 900.0,
            ..PoleConfig::default()
        };
        assert!(bad.validate().is_err());
        assert!(PoleConfig::default().validate().is_ok());
    }
}
