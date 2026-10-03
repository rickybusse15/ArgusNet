//! Float comparison for the parity fixtures, exact where it can be.
//!
//! Every parity fixture is a capture of the Python reference, and Python's
//! `math.sin`, `math.exp`, `np.linalg.norm` and friends call the platform's
//! maths library. So does Rust's `f64::sin`. On the platform a fixture was
//! captured on the two agree bit for bit, and these tests compare bits. On
//! another platform (glibc against Apple's libm, say) the same expression can
//! round one unit in the last place differently, so a bit-exact comparison
//! there tests the libm, not the port.
//!
//! Each fixture records where it was captured (`_captured_on`, as
//! `"<os>-<arch>"`). On that platform comparisons stay exact. Elsewhere a
//! value passes within `max_ulps` units in the last place, or within
//! `abs_floor` of the reference for values near zero, where ULPs are
//! meaningless. This is the documented cross-platform exception of ADR-006
//! (see `docs/migration/PARITY_PORTABILITY.md`); it never applies on the
//! capture platform.

#![allow(dead_code)]

use serde_json::Value;

/// The cross-platform bound. Measured on linux-x86_64 (glibc 2.39) against
/// fixtures captured on macos-aarch64: every value in every suite, whole runs
/// included, is within 1 ULP. 2 leaves one unit for other libm builds; a
/// failure beyond it is a real divergence, not rounding.
pub const MAX_ULPS: u64 = 2;
/// Near zero a ULP is meaningless; 1e-12 (metres, m/s, or unitless) is the
/// builder's own height tolerance.
pub const ABS_FLOOR: f64 = 1e-12;

/// [`Parity::for_fixture`] with the suite-wide bound.
pub fn parity_of(fixture: &Value) -> Parity {
    Parity::for_fixture(fixture, MAX_ULPS, ABS_FLOOR)
}

/// `"<os>-<arch>"` of the running test, as fixtures record it.
pub fn this_platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Parity {
    /// Bit for bit: the fixture was captured on this platform.
    Exact,
    /// Within `max_ulps`, or within `abs_floor` absolutely.
    Bounded { max_ulps: u64, abs_floor: f64 },
}

impl Parity {
    /// Exact if `fixture["_captured_on"]` names this platform, otherwise
    /// the given bound. A fixture without the tag is a capture-script bug
    /// and fails loudly rather than silently choosing a mode.
    pub fn for_fixture(fixture: &Value, max_ulps: u64, abs_floor: f64) -> Self {
        let captured_on = fixture["_captured_on"]
            .as_str()
            .expect("fixture lacks `_captured_on`; recapture it with its script");
        if captured_on == this_platform() {
            Parity::Exact
        } else {
            Parity::Bounded {
                max_ulps,
                abs_floor,
            }
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, Parity::Exact)
    }

    pub fn same(&self, got: f64, want: f64) -> bool {
        match *self {
            Parity::Exact => got.to_bits() == want.to_bits(),
            Parity::Bounded {
                max_ulps,
                abs_floor,
            } => {
                if got.to_bits() == want.to_bits() {
                    return true;
                }
                if got.is_nan() || want.is_nan() || got.is_infinite() || want.is_infinite() {
                    return false;
                }
                (got - want).abs() <= abs_floor || ulps_apart(got, want) <= max_ulps
            }
        }
    }

    pub fn same_all(&self, got: &[f64], want: &[f64]) -> bool {
        got.len() == want.len() && got.iter().zip(want).all(|(g, w)| self.same(*g, *w))
    }
}

/// A recorded oracle: answers the reference gave, keyed by the exact float
/// arguments it was asked with.
///
/// On the capture platform a lookup is exact, so a port that asks a question
/// the reference never asked fails, which is the point of the oracle. Off it,
/// a port's arguments can sit a few ULPs from the recorded ones, so a miss
/// falls back to the recorded key closest within the parity bound. A key
/// component may be `NaN` to stand for "not given" (an absent optional
/// argument); it matches only itself.
pub struct ReplayTable<V> {
    parity: Parity,
    exact: std::collections::HashMap<Vec<u64>, usize>,
    entries: Vec<(Vec<f64>, V)>,
}

impl<V> ReplayTable<V> {
    pub fn new(parity: Parity) -> Self {
        Self {
            parity,
            exact: std::collections::HashMap::new(),
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, key: Vec<f64>, value: V) {
        let bits = key.iter().map(|v| v.to_bits()).collect();
        self.exact.insert(bits, self.entries.len());
        self.entries.push((key, value));
    }

    pub fn get(&self, query: &[f64]) -> Option<&V> {
        let bits: Vec<u64> = query.iter().map(|v| v.to_bits()).collect();
        if let Some(&index) = self.exact.get(&bits) {
            return Some(&self.entries[index].1);
        }
        if self.parity.is_exact() {
            return None;
        }
        self.entries
            .iter()
            .filter(|(key, _)| self.parity.same_all(query, key))
            .min_by_key(|(key, _)| {
                key.iter()
                    .zip(query)
                    .map(|(k, q)| ulps_apart(*k, *q))
                    .fold(0u64, u64::saturating_add)
            })
            .map(|(_, value)| value)
    }
}

/// Units in the last place between two finite doubles; adjacent values are
/// one apart, and `-0.0` and `0.0` are zero apart.
pub fn ulps_apart(a: f64, b: f64) -> u64 {
    fn ordered(x: f64) -> i128 {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            -i128::from(bits & i64::MAX)
        } else {
            i128::from(bits)
        }
    }
    (ordered(a) - ordered(b)).unsigned_abs() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_doubles_are_one_ulp_apart() {
        let x = 0.7802705479997403_f64;
        let next = f64::from_bits(x.to_bits() + 1);
        assert_eq!(ulps_apart(x, next), 1);
        assert_eq!(ulps_apart(-0.0, 0.0), 0);
        assert_eq!(
            ulps_apart(-f64::MIN_POSITIVE, f64::MIN_POSITIVE),
            2 * MIN_POSITIVE_BITS
        );
    }

    const MIN_POSITIVE_BITS: u64 = 0x0010_0000_0000_0000;

    #[test]
    fn exact_mode_is_bit_for_bit_and_bounded_mode_is_not_lenient_beyond_its_bound() {
        let x = 1.5946142049572598_f64;
        let one_off = f64::from_bits(x.to_bits() - 1);
        assert!(!Parity::Exact.same(one_off, x));
        let bounded = Parity::Bounded {
            max_ulps: 4,
            abs_floor: 0.0,
        };
        assert!(bounded.same(one_off, x));
        assert!(!bounded.same(f64::from_bits(x.to_bits() + 5), x));
        assert!(!bounded.same(f64::NAN, x));
        let floored = Parity::Bounded {
            max_ulps: 0,
            abs_floor: 1e-12,
        };
        assert!(floored.same(1e-17, 0.0));
        assert!(!floored.same(2e-12, 0.0));
    }

    #[test]
    fn a_replay_table_is_exact_on_the_capture_platform_and_bounded_elsewhere() {
        let x = 150.3069604358496_f64;
        let shifted = f64::from_bits(x.to_bits() + 1);
        let mut exact = ReplayTable::new(Parity::Exact);
        exact.insert(vec![x, f64::NAN], 1);
        assert_eq!(exact.get(&[x, f64::NAN]), Some(&1));
        assert_eq!(exact.get(&[shifted, f64::NAN]), None);

        let mut bounded = ReplayTable::new(Parity::Bounded {
            max_ulps: 4,
            abs_floor: 0.0,
        });
        bounded.insert(vec![x, 2.0], 1);
        bounded.insert(vec![f64::from_bits(x.to_bits() + 3), 2.0], 2);
        assert_eq!(bounded.get(&[shifted, 2.0]), Some(&1), "the closer key");
        assert_eq!(
            bounded.get(&[f64::from_bits(x.to_bits() + 4), 2.0]),
            Some(&2)
        );
        assert_eq!(bounded.get(&[f64::from_bits(x.to_bits() + 9), 2.0]), None);
        assert_eq!(bounded.get(&[x, 3.0]), None);
    }

    #[test]
    fn a_fixture_captured_here_is_exact() {
        let here = serde_json::json!({ "_captured_on": this_platform() });
        assert!(Parity::for_fixture(&here, 4, 0.0).is_exact());
        let elsewhere = serde_json::json!({ "_captured_on": "nowhere-none" });
        assert!(!Parity::for_fixture(&elsewhere, 4, 0.0).is_exact());
    }
}
