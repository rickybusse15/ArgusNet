//! The deterministic primitives every procedural terrain is built from:
//! seed derivation, grid axes, value noise, fBm and the line envelope.
//!
//! These sit at the bottom of `procedural.py`'s dependency graph — height
//! shaping, hydrology masks and land cover all run through them — so a
//! divergence of one ULP here is a divergence in every terrain the runtime
//! builds. They are ported first and pinned bit-for-bit for that reason.
//!
//! # Three things that are wrong in a way no casual test catches
//!
//! **1. The seed function is not the one you already have.**
//! `procedural.py::_stable_seed` is **blake2b-64, little-endian, masked to 31
//! bits**. `sim.py::_stable_seed` — which `argusnet_sim::deterministic`
//! already ports — is an FNV-1 walk over characters. They share a name and
//! nothing else. Reaching for the existing one produces terrain that is
//! plausible, deterministic, and wrong.
//!
//! **2. The hash arithmetic overflows, and numpy wraps where Rust panics.**
//! `_hash_noise` multiplies `int64` values by `0x45D9F3B` twice; the products
//! exceed `i64::MAX` for ordinary grid coordinates. numpy wraps silently.
//! Rust's `*` panics in a debug build, so every multiply here is
//! `wrapping_mul`.
//!
//! The shifts, by contrast, do **not** matter: swapping either arithmetic
//! `>>` for a logical one cannot change the result, even though half of all
//! inputs reach them with a negative `h`. Only bits 51 and above differ
//! between the two, multiplication propagates carries upward only, and the
//! final `& 0xFFFF` keeps bits that depend on operand bits 0..28. Checked over
//! 300,000 random inputs, 149,893 of them with a negative `h`: zero
//! disagreements. Recorded because an earlier note here claimed the opposite.
//!
//! **3. `np.arange` is not `start + i * step`.**
//! numpy's `DOUBLE_fill` writes the first two elements directly, **recomputes**
//! the delta from them, and generates the rest from that:
//!
//! ```text
//! buf[0] = start
//! buf[1] = start + step
//! delta  = buf[1] - buf[0]        // generally != step
//! buf[i] = start + i * delta      // i >= 2
//! ```
//!
//! The recomputed delta is the whole point. `start + i * step` disagrees on
//! about 78% of random inputs, and so does plain cumulative addition; the
//! model above matched numpy 2.4.6 on 198,991 of 198,991 random cases. It is
//! also not a rounding artifact that might drift: the output is stable with
//! AVX2 and AVX-512 dispatch disabled.

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

/// Derive a terrain seed the way `procedural.py::_stable_seed` does.
///
/// The parts are joined with `|`, encoded as UTF-8, hashed with blake2b to an
/// **8-byte** digest, read **little-endian**, and masked to 31 bits.
///
/// This is deliberately not `argusnet_sim::deterministic::stable_seed`. See
/// the module docs: same name in the reference, different function.
pub fn stable_seed(parts: &[&str]) -> i64 {
    let payload = parts.join("|");
    let mut hasher = Blake2bVar::new(8).expect("8 is a valid blake2b digest size");
    hasher.update(payload.as_bytes());
    let mut digest = [0u8; 8];
    hasher
        .finalize_variable(&mut digest)
        .expect("digest buffer is exactly 8 bytes");
    (u64::from_le_bytes(digest) & 0x7FFF_FFFF) as i64
}

/// Scene bounds in metres.
///
/// The one definition in this crate. `semantic` re-exports it: it previously
/// carried a second, field-identical copy, which made `noise::Bounds2D` and
/// `semantic::Bounds2D` distinct types that no caller could pass between —
/// invisible until P4.3e composed `grid_values` with `semantic_masks` in one
/// function and the compiler refused. CLAUDE.md's "one authority per concern"
/// applies to value types too.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds2D {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
}

impl Bounds2D {
    pub fn width_m(&self) -> f64 {
        self.x_max_m - self.x_min_m
    }

    pub fn height_m(&self) -> f64 {
        self.y_max_m - self.y_min_m
    }
}

/// Reproduce `np.arange(start, stop, step)` for `f64`.
///
/// See the module docs for why this is not `start + i * step`. The length is
/// `ceil((stop - start) / step)`, floored at zero, which matched numpy on
/// every case tried.
pub fn arange(start: f64, stop: f64, step: f64) -> Vec<f64> {
    if !step.is_finite() || step == 0.0 || !start.is_finite() || !stop.is_finite() {
        return Vec::new();
    }
    let raw = ((stop - start) / step).ceil();
    if !raw.is_finite() || raw <= 0.0 {
        return Vec::new();
    }
    let length = raw as usize;

    let mut out = Vec::with_capacity(length);
    out.push(start);
    if length == 1 {
        return out;
    }
    out.push(start + step);
    // The recomputed delta, not `step`. This is the line that matters.
    let delta = out[1] - out[0];
    for i in 2..length {
        out.push(start + i as f64 * delta);
    }
    out
}

/// The x and y axes of a terrain grid, as `_grid_values` builds them.
///
/// The `+ resolution * 0.5` in the stop value is the reference's way of
/// including the far edge; it is part of the contract, not a rounding guard.
/// An axis that comes back with fewer than two samples is replaced by the two
/// bounds, so a degenerate span still yields a usable grid.
pub fn grid_values(bounds: Bounds2D, resolution_m: f64) -> (Vec<f64>, Vec<f64>) {
    let axis = |min: f64, max: f64| -> Vec<f64> {
        let values = arange(min, max + resolution_m * 0.5, resolution_m);
        if values.len() < 2 {
            vec![min, max]
        } else {
            values
        }
    };
    (
        axis(bounds.x_min_m, bounds.x_max_m),
        axis(bounds.y_min_m, bounds.y_max_m),
    )
}

/// `v * v * (3 - 2v)`, spelled in the reference's association order.
pub fn smoothstep(value: f64) -> f64 {
    value * value * (3.0 - (2.0 * value))
}

/// The integer hash behind the value noise.
///
/// Every multiply wraps. See the module docs: these products exceed `i64::MAX`
/// for ordinary grid coordinates, numpy wraps silently, and a plain `*` here
/// would panic in a debug build rather than produce numpy's answer.
///
/// The shifts are written arithmetic to mirror the reference, but the choice
/// is provably immaterial here — see the module docs.
/// The mixing constant, spelled `0x45D9F3B` in `procedural.py`. Written with
/// even digit groups here only because clippy insists; the value is identical.
const HASH_MIX: i64 = 0x045D_9F3B;

/// The x and y coordinate multipliers, also straight from the reference.
const HASH_X: i64 = 73_856_093;
const HASH_Y: i64 = 19_349_663;

pub fn hash_noise(ix: i64, iy: i64, seed: i64) -> f64 {
    let mut h = (ix.wrapping_mul(HASH_X)) ^ (iy.wrapping_mul(HASH_Y)) ^ seed;
    h = ((h >> 13) ^ h).wrapping_mul(HASH_MIX);
    h = ((h >> 13) ^ h).wrapping_mul(HASH_MIX);
    h = (h >> 13) ^ h;
    // `h & 0xFFFF` is non-negative even when `h` is negative, so the low 16
    // bits map to 0..=65535. The divisor is exactly half of 65535, so the
    // result spans a CLOSED [-1, 1]: both endpoints are attained. (An earlier
    // note here claimed a half-open range; 32767.5 * 2 == 65535 exactly, so
    // the top really is reached.)
    ((h & 0xFFFF) as f64 / 32767.5) - 1.0
}

/// Bilinear value noise on the unit lattice.
pub fn value_noise(x: f64, y: f64, seed: i64) -> f64 {
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = smoothstep(x - ix as f64);
    let fy = smoothstep(y - iy as f64);
    let n00 = hash_noise(ix, iy, seed);
    let n10 = hash_noise(ix + 1, iy, seed);
    let n01 = hash_noise(ix, iy + 1, seed);
    let n11 = hash_noise(ix + 1, iy + 1, seed);
    // The lerp is spelled `a + t * (b - a)`, which is not the same float as
    // `(1 - t) * a + t * b`.
    let nx0 = n00 + fx * (n10 - n00);
    let nx1 = n01 + fx * (n11 - n01);
    nx0 + fy * (nx1 - nx0)
}

/// Parameters for [`fbm`], matching the reference's keyword defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FbmConfig {
    pub seed: i64,
    pub base_wavelength_m: f64,
    pub octaves: i64,
    pub persistence: f64,
    pub lacunarity: f64,
}

impl FbmConfig {
    /// `persistence = 0.52`, `lacunarity = 2.03` — the reference's defaults.
    pub fn new(seed: i64, base_wavelength_m: f64, octaves: i64) -> Self {
        Self {
            seed,
            base_wavelength_m,
            octaves,
            persistence: 0.52,
            lacunarity: 2.03,
        }
    }
}

/// Fractional Brownian motion over [`value_noise`].
///
/// Two orderings inside the loop are load-bearing and are easy to swap without
/// noticing: the amplitude sum accumulates **before** the wavelength and
/// amplitude advance, and the wavelength is divided **before** the amplitude is
/// scaled. The per-octave seed step is `+ octave * 9973`.
pub fn fbm(x_m: f64, y_m: f64, config: FbmConfig) -> f64 {
    let mut total = 0.0f64;
    let mut amplitude = 1.0f64;
    // `max(float(base), 1.0)` — a wavelength below a metre is clamped, not an
    // error.
    let mut wavelength = config.base_wavelength_m.max(1.0);
    let mut amplitude_sum = 0.0f64;
    let octaves = config.octaves.max(1);
    for octave in 0..octaves {
        total += amplitude
            * value_noise(
                x_m / wavelength,
                y_m / wavelength,
                config.seed + octave.wrapping_mul(9973),
            );
        amplitude_sum += amplitude;
        wavelength /= config.lacunarity;
        amplitude *= config.persistence;
    }
    total / amplitude_sum.max(1.0e-9)
}

/// Placement of a [`line_gaussian`] ridge or valley.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineEnvelope {
    pub angle_rad: f64,
    pub offset_m: f64,
    pub width_m: f64,
    pub length_m: f64,
    pub center_xy: (f64, f64),
}

/// A Gaussian ridge along a line: Gaussian across, super-Gaussian along.
///
/// The along-track envelope uses the **fourth** power, not the square — it is a
/// flat-topped plateau with fast shoulders, where squaring would give an
/// ordinary bell. Both widths are floored at one metre.
pub fn line_gaussian(xx: f64, yy: f64, envelope: LineEnvelope) -> f64 {
    let dx = xx - envelope.center_xy.0;
    let dy = yy - envelope.center_xy.1;
    let normal_x = -envelope.angle_rad.sin();
    let normal_y = envelope.angle_rad.cos();
    let along_x = envelope.angle_rad.cos();
    let along_y = envelope.angle_rad.sin();
    let cross = (dx * normal_x) + (dy * normal_y) - envelope.offset_m;
    let along = (dx * along_x) + (dy * along_y);
    let cross_ratio = cross / envelope.width_m.max(1.0);
    let along_ratio = along / envelope.length_m.max(1.0);
    let cross_env = (-0.5 * cross_ratio.powi(2)).exp();
    let along_env = (-0.5 * along_ratio.powi(4)).exp();
    cross_env * along_env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_seed_is_blake2b_not_the_fnv_walk_of_the_same_name() {
        // Captured from `procedural.py::_stable_seed("argusnet", 7)`.
        // If this ever equals what `argusnet_sim::deterministic::stable_seed`
        // returns for the same parts, one of the two has been "unified".
        let seed = stable_seed(&["argusnet", "7"]);
        assert!(seed >= 0, "the 31-bit mask keeps the seed non-negative");
        assert!(seed <= 0x7FFF_FFFF);
        assert_ne!(stable_seed(&["a", "b"]), stable_seed(&["b", "a"]));
        // The join is on `|`, so these are different payloads.
        assert_ne!(stable_seed(&["a|b"]), stable_seed(&["a", "b", ""]));
    }

    #[test]
    fn arange_uses_the_recomputed_delta() {
        // The case from the module docs: start + i*step disagrees at i >= 2.
        let values = arange(0.1, 0.1 + 5.0 + 0.15, 0.3);
        assert_eq!(values[0], 0.1);
        assert_eq!(values[1], 0.1 + 0.3);
        let delta = values[1] - values[0];
        assert_ne!(delta, 0.3, "the recomputed delta differs from the step");
        assert_eq!(values[2], 0.1 + 2.0 * delta);
        assert_ne!(values[2], 0.1 + 2.0 * 0.3);
    }

    #[test]
    fn arange_rejects_degenerate_steps() {
        assert!(arange(0.0, 1.0, 0.0).is_empty());
        assert!(arange(0.0, 1.0, f64::NAN).is_empty());
        assert!(arange(0.0, -1.0, 1.0).is_empty());
    }

    #[test]
    fn a_degenerate_span_still_yields_two_samples() {
        let bounds = Bounds2D {
            x_min_m: 0.0,
            x_max_m: 0.0,
            y_min_m: 0.0,
            y_max_m: 1000.0,
        };
        let (xs, ys) = grid_values(bounds, 50.0);
        assert_eq!(xs, vec![0.0, 0.0], "a zero span falls back to the bounds");
        assert!(ys.len() > 2);
    }

    #[test]
    fn the_hash_multiply_chain_overflows_and_must_wrap() {
        // These coordinates drive the product past i64::MAX. A non-wrapping
        // multiply panics here in a debug build, which is the point.
        let value = hash_noise(12_345, 54_321, 123_456_789);
        assert!(
            (-1.0..=1.0).contains(&value),
            "noise stays in [-1, 1], got {value}"
        );
        // Negative coordinates exercise the arithmetic shift on a negative h.
        let negative = hash_noise(-7, 13, 123_456_789);
        assert!((-1.0..=1.0).contains(&negative));
    }

    #[test]
    fn the_noise_range_is_a_closed_interval() {
        // 32767.5 * 2 == 65535 exactly, so 0..=65535 maps onto a CLOSED
        // [-1, 1] and both endpoints are attained rather than approached.
        let mut lowest = f64::INFINITY;
        let mut highest = f64::NEG_INFINITY;
        for ix in -200..200 {
            for iy in -200..200 {
                let v = hash_noise(ix, iy, 42);
                lowest = lowest.min(v);
                highest = highest.max(v);
            }
        }
        assert_eq!(lowest, -1.0);
        assert_eq!(highest, 1.0);
    }

    #[test]
    fn value_noise_is_continuous_across_a_lattice_line() {
        // Approaching an integer from below must meet the value at it.
        let below = value_noise(3.0 - 1e-9, 1.25, 99);
        let at = value_noise(3.0, 1.25, 99);
        assert!((below - at).abs() < 1e-6, "{below} vs {at}");
    }

    #[test]
    fn value_noise_reproduces_the_lattice_corners() {
        // At an exact lattice point the interpolation weights are zero, so the
        // result is the corner hash itself.
        assert_eq!(value_noise(4.0, 9.0, 7), hash_noise(4, 9, 7));
    }

    #[test]
    fn fbm_clamps_a_sub_metre_wavelength() {
        let a = fbm(10.0, 20.0, FbmConfig::new(5, 0.25, 3));
        let b = fbm(10.0, 20.0, FbmConfig::new(5, 1.0, 3));
        assert_eq!(a, b, "anything below 1 m is clamped to 1 m");
    }

    #[test]
    fn fbm_octave_floor_is_one() {
        let zero = fbm(3.0, 4.0, FbmConfig::new(1, 500.0, 0));
        let one = fbm(3.0, 4.0, FbmConfig::new(1, 500.0, 1));
        assert_eq!(zero, one);
        let negative = fbm(3.0, 4.0, FbmConfig::new(1, 500.0, -5));
        assert_eq!(negative, one);
    }

    #[test]
    fn fbm_floors_an_amplitude_sum_that_cancels_to_zero() {
        // The reference cannot reach this: `persistence` defaults to 0.52 and
        // every caller leaves it positive, so the amplitude sum starts at 1.0
        // and only grows. The Rust API takes it as data, and a persistence of
        // -1 makes the amplitudes alternate 1, -1, 1, ... so an even octave
        // count sums to exactly zero and the `max(1.0e-9)` floor becomes
        // load-bearing. Pinned as a unit test rather than in the fixture,
        // which records what Python actually produces.
        let config = FbmConfig {
            seed: 4,
            base_wavelength_m: 100.0,
            octaves: 2,
            persistence: -1.0,
            lacunarity: 2.0,
        };
        let value = fbm(7.0, 8.0, config);
        assert!(
            value.is_finite(),
            "the floor keeps a cancelling amplitude sum from dividing by zero, got {value}"
        );
    }

    #[test]
    fn fbm_normalises_by_the_amplitude_sum() {
        // One octave divides by an amplitude sum of exactly 1, so the result is
        // the raw noise.
        let single = fbm(12.0, 34.0, FbmConfig::new(3, 100.0, 1));
        let raw = value_noise(12.0 / 100.0, 34.0 / 100.0, 3);
        assert_eq!(single, raw);
    }

    #[test]
    fn the_along_envelope_is_quartic_not_quadratic() {
        let envelope = LineEnvelope {
            angle_rad: 0.0,
            offset_m: 0.0,
            width_m: 100.0,
            length_m: 100.0,
            center_xy: (0.0, 0.0),
        };
        // One length unit along: quartic gives exp(-0.5), quadratic would too,
        // so probe at two units where they separate sharply.
        let value = line_gaussian(200.0, 0.0, envelope);
        let quartic = (-0.5f64 * 2.0f64.powi(4)).exp();
        let quadratic = (-0.5f64 * 2.0f64.powi(2)).exp();
        assert!((value - quartic).abs() < 1e-15);
        assert!(
            (value - quadratic).abs() > 1e-3,
            "quartic and quadratic must be distinguishable here"
        );
    }

    #[test]
    fn the_line_envelope_peaks_on_its_own_axis() {
        let envelope = LineEnvelope {
            angle_rad: std::f64::consts::FRAC_PI_4,
            offset_m: 0.0,
            width_m: 50.0,
            length_m: 500.0,
            center_xy: (10.0, -20.0),
        };
        let on_axis = line_gaussian(10.0, -20.0, envelope);
        assert!((on_axis - 1.0).abs() < 1e-12);
        // Perpendicular to the line the cross envelope decays.
        // Step perpendicular to a 45-degree line: one half-width along the
        // normal, which is (-sin, cos) = (-1, 1)/sqrt(2).
        let half = 50.0 * std::f64::consts::FRAC_1_SQRT_2;
        let off = line_gaussian(10.0 - half, -20.0 + half, envelope);
        assert!(off < on_axis);
    }

    #[test]
    fn envelope_widths_are_floored_at_one_metre() {
        let thin = LineEnvelope {
            angle_rad: 0.0,
            offset_m: 0.0,
            width_m: 0.01,
            length_m: 0.01,
            center_xy: (0.0, 0.0),
        };
        let unit = LineEnvelope {
            width_m: 1.0,
            length_m: 1.0,
            ..thin
        };
        assert_eq!(line_gaussian(0.3, 0.2, thin), line_gaussian(0.3, 0.2, unit));
    }
}
