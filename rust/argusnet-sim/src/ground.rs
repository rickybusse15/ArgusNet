//! Where a straight flight comes closest to the ground, exactly.
//!
//! The height under a straight segment of a grid terrain is a quadratic in
//! each cell it crosses ([`TiledTerrain::height_along`]), and a flight whose
//! height changes linearly along the segment is a quadratic above it too, so
//! the lowest point and the first point within a clearance are solved for,
//! not sampled: no ridge between two samples is missed. A terrain with no
//! grid (an analytic surface in a test) is fitted every
//! [`ANALYTIC_PIECE_M`] instead.
//!
//! [`TiledTerrain::height_along`]: argusnet_world::tiled::TiledTerrain::height_along

use argusnet_world::tiled::HeightPiece;

use crate::observe::TerrainHeight;

/// How long each fitted piece of a terrain with no grid is, metres.
pub const ANALYTIC_PIECE_M: f64 = 0.25;

/// The ground under `from` to `to`: each piece's height, or the floor's
/// (a grid's height is the larger of the two, so a margin is the smaller).
fn pieces(terrain: &dyn TerrainHeight, from: [f64; 3], to: [f64; 3]) -> Vec<HeightPiece> {
    let (a, b) = ([from[0], from[1]], [to[0], to[1]]);
    if let Some((mut pieces, floor_m)) = terrain.height_along(a, b) {
        let floors: Vec<HeightPiece> = pieces
            .iter()
            .map(|piece| HeightPiece {
                coefficients: [floor_m, 0.0, 0.0],
                ..*piece
            })
            .collect();
        pieces.extend(floors);
        return pieces;
    }
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    let count = (length / ANALYTIC_PIECE_M).ceil().max(1.0) as usize;
    (0..count)
        .map(|k| {
            let (t0, t1) = (k as f64 / count as f64, (k + 1) as f64 / count as f64);
            let h = |s: f64| {
                let t = t0 + (t1 - t0) * s;
                terrain.height_at(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
            };
            let (y0, ym, y1) = (h(0.0), h(0.5), h(1.0));
            let c2 = 2.0 * (y0 - 2.0 * ym + y1);
            HeightPiece {
                t0,
                t1,
                coefficients: [y0, y1 - y0 - c2, c2],
            }
        })
        .collect()
}

/// Over one piece, the flight's height above the ground less `clearance_m`
/// as `[a, b, c]`: `a + b·s + c·s²` with `s` 0 to 1 across the piece.
fn margin(piece: &HeightPiece, from: [f64; 3], to: [f64; 3], clearance_m: f64) -> [f64; 3] {
    let z0 = from[2] + (to[2] - from[2]) * piece.t0;
    let dz = (to[2] - from[2]) * (piece.t1 - piece.t0);
    let [c0, c1, c2] = piece.coefficients;
    [z0 - c0 - clearance_m, dz - c1, -c2]
}

fn value([a, b, c]: [f64; 3], s: f64) -> f64 {
    a + (b + c * s) * s
}

/// The lowest of `[a, b, c]` over `s` in 0..=1, and where.
fn lowest(q: [f64; 3]) -> (f64, f64) {
    let mut best = (value(q, 0.0), 0.0);
    let mut consider = |s: f64| {
        let v = value(q, s);
        if v < best.0 {
            best = (v, s);
        }
    };
    consider(1.0);
    if q[2] > 0.0 {
        let vertex = -q[1] / (2.0 * q[2]);
        if vertex > 0.0 && vertex < 1.0 {
            consider(vertex);
        }
    }
    best
}

/// How far the flight from `from` to `to` (its height changing linearly)
/// comes above the ground at its lowest, and the fraction along it where.
pub fn lowest_margin(terrain: &dyn TerrainHeight, from: [f64; 3], to: [f64; 3]) -> (f64, f64) {
    pieces(terrain, from, to)
        .iter()
        .map(|piece| {
            let (m, s) = lowest(margin(piece, from, to, 0.0));
            (m, piece.t0 + (piece.t1 - piece.t0) * s)
        })
        .fold((f64::INFINITY, 0.0), |best, next| {
            if next.0 < best.0 {
                next
            } else {
                best
            }
        })
}

/// The first fraction along the flight from `from` to `to` at which it is
/// less than `clearance_m` above the ground, or `None` if it never is.
pub fn first_within(
    terrain: &dyn TerrainHeight,
    from: [f64; 3],
    to: [f64; 3],
    clearance_m: f64,
) -> Option<f64> {
    pieces(terrain, from, to)
        .iter()
        .filter_map(|piece| {
            let q = margin(piece, from, to, clearance_m);
            if lowest(q).0 >= 0.0 {
                return None;
            }
            let s = if q[0] < 0.0 {
                0.0
            } else {
                // From at or above the clearance to below it: the first root.
                let [a, b, c] = q;
                if c.abs() < 1e-12 {
                    -a / b
                } else {
                    let root = (b * b - 4.0 * a * c).max(0.0).sqrt();
                    [(-b - root) / (2.0 * c), (-b + root) / (2.0 * c)]
                        .into_iter()
                        .filter(|s| (0.0..=1.0).contains(s))
                        .fold(1.0, f64::min)
                }
            };
            Some(piece.t0 + (piece.t1 - piece.t0) * s.clamp(0.0, 1.0))
        })
        .reduce(f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_world::SharedTerrain;
    use argusnet_types::Bounds2d;
    use argusnet_world::tiled::{Tile, TiledTerrain};
    use std::sync::Arc;

    /// One tile of 4 x 4 cells of 10 m, flat at 0 but for a node 50 m high
    /// at (20, 20): a peak no sample at whole multiples of 3 m lands on.
    fn peaked() -> SharedTerrain {
        let heights_m = (0..5)
            .map(|row| {
                (0..5)
                    .map(|col| if (row, col) == (2, 2) { 50.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let tile = Tile {
            tx: 0,
            ty: 0,
            lod: 0,
            x_min_m: 0.0,
            y_min_m: 0.0,
            cell_size_m: 10.0,
            heights_m,
            min_pyramid: Vec::new(),
            max_pyramid: Vec::new(),
        };
        let terrain = TiledTerrain::new(
            Bounds2d::new(0.0, 40.0, 0.0, 40.0),
            4,
            10.0,
            -5.0,
            vec![tile],
        )
        .unwrap();
        SharedTerrain::new(Arc::new(terrain), 10.0)
    }

    /// Across the peak at 45 m: it comes 5 m over it, at the peak, and is
    /// first within 8 m of the ground just before.
    #[test]
    fn the_peak_between_samples_is_found() {
        let terrain = peaked();
        let (from, to) = ([1.0, 20.0, 55.0], [39.0, 20.0, 55.0]);
        let (low, t) = lowest_margin(&terrain, from, to);
        assert!((low - 5.0).abs() < 1e-9, "{low}");
        assert!((t - 0.5).abs() < 1e-9, "{t}");
        let first = first_within(&terrain, from, to, 8.0).unwrap();
        // Ground rises 5 m per metre toward the peak: 47 m at 0.6 m short.
        let x = from[0] + (to[0] - from[0]) * first;
        assert!((x - 19.4).abs() < 1e-9, "{x}");
        assert_eq!(first_within(&terrain, from, to, 4.0), None);
        // Sampled every 3 m from x = 1, the peak at x = 20 is missed by a
        // metre either side, and the flight looks twice as clear as it is.
        let sampled = (0..=12)
            .map(|k| 55.0 - TerrainHeight::height_at(&terrain, 1.0 + 3.0 * k as f64, 20.0))
            .fold(f64::INFINITY, f64::min);
        assert!(sampled > 9.0, "{sampled}");
    }

    /// A climbing flight over flat ground: lowest at its start.
    #[test]
    fn a_climb_is_lowest_where_it_starts() {
        let terrain = peaked();
        let (low, t) = lowest_margin(&terrain, [1.0, 1.0, 10.0], [9.0, 1.0, 30.0]);
        assert!((low - 10.0).abs() < 1e-9);
        assert_eq!(t, 0.0);
    }
}
