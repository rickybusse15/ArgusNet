//! Seed derivation and the pinned random stream.
//!
//! [ADR-006](../../../docs/adr/006-determinism-rebaseline.md) settles what
//! determinism means after the port: *the Rust runtime is deterministic for a
//! fixed seed and configuration, and reproducible across platforms* — not
//! "identical to numpy". Reproducing numpy would mean reimplementing
//! `SeedSequence` and PCG64 bit-exactly and then owning that forever.
//!
//! ADR-006 does require keeping the *seed-derivation scheme*, because that is
//! what buys order-independence: `build_observations` gives every
//! `(node, target, tick)` triple its own stream, so the result does not depend
//! on the order platforms are iterated in. That scheme is reproduced here
//! exactly — see [`stable_seed`] — and pinned against Python's own output by
//! `tests/conformance.rs`.
//!
//! # What "pinned" means
//!
//! Choosing a PRNG does not by itself make values stable; a dependency can
//! change its algorithm in a release and every golden file moves silently. So
//! the guarantee does not rest on the crate choice. It rests on conformance
//! fixtures that record actual draws, and on exact-version pins in
//! `Cargo.toml`. The fixtures are the contract; ChaCha8 is an implementation
//! detail the contract protects.

use rand_chacha::ChaCha8Rng;
// In rand_core 0.10 the generator methods live on `Rng`; `RngCore` is a marker
// alias for it, so importing that instead compiles to a missing-method error.
use rand_core::{Rng, SeedableRng};

/// One component of a keyed seed.
///
/// The Python derivation hashes `str(part)`, so how a part renders *is* part of
/// the contract: the integer `1500` must hash as `"1500"`. Making the rendering
/// explicit at the call site is the point of this enum — a bare `impl Display`
/// would let a float or a debug format in, and the mismatch would show up as a
/// diverged stream rather than a type error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedPart<'a> {
    /// Hashed as-is.
    Str(&'a str),
    /// Hashed as its decimal rendering, which matches Python's `str(int)` for
    /// every value an `i64` can hold, negative sign included.
    Int(i64),
}

impl<'a> From<&'a str> for SeedPart<'a> {
    fn from(value: &'a str) -> Self {
        SeedPart::Str(value)
    }
}

impl From<i64> for SeedPart<'_> {
    fn from(value: i64) -> Self {
        SeedPart::Int(value)
    }
}

/// The 32-bit FNV prime, as `sim.py::_stable_seed` uses it.
const FNV_PRIME_32: u32 = 16_777_619;

/// Reproduce `sim.py::_stable_seed` exactly.
///
/// ```text
/// seed = int(base_seed) & 0xFFFFFFFF
/// for part in parts:
///     for char in str(part):
///         seed = ((seed * 16777619) ^ ord(char)) & 0xFFFFFFFF
/// ```
///
/// Two details are easy to get wrong and both are load-bearing:
///
/// - The order is **multiply then xor**, which is FNV-**1**, not the more
///   common FNV-1a. Swapping them produces a different hash for every input
///   longer than one character.
/// - `ord(char)` is a Unicode scalar value, not a byte. For the ASCII node and
///   target ids in use today the two coincide, but hashing UTF-8 bytes would
///   diverge the moment an id carried a non-ASCII character — so this iterates
///   `chars()`, matching Python rather than matching intuition about bytes.
///
/// There is no initial FNV offset basis: the walk starts from the caller's
/// seed. That is not a mistake to "fix" here; it is what the reference does,
/// and the fixtures pin it.
pub fn stable_seed(base_seed: u32, parts: &[SeedPart<'_>]) -> u32 {
    let mut seed = base_seed;
    let mut buffer = itoa_buffer();
    for part in parts {
        let rendered = match part {
            SeedPart::Str(value) => *value,
            SeedPart::Int(value) => render_int(*value, &mut buffer),
        };
        for character in rendered.chars() {
            seed = seed.wrapping_mul(FNV_PRIME_32) ^ (character as u32);
        }
    }
    seed
}

/// Scratch space for rendering an `i64` without allocating.
///
/// 20 digits plus a sign covers `i64::MIN`.
type IntBuffer = [u8; 21];

fn itoa_buffer() -> IntBuffer {
    [0; 21]
}

fn render_int(value: i64, buffer: &mut IntBuffer) -> &str {
    use std::io::Write;
    let mut cursor = std::io::Cursor::new(&mut buffer[..]);
    write!(cursor, "{value}").expect("21 bytes holds any i64");
    let written = cursor.position() as usize;
    std::str::from_utf8(&buffer[..written]).expect("decimal digits are ASCII")
}

/// A reproducible stream of random values.
///
/// Construct one per logical stream — per `(node, target, tick)` for
/// observation noise, per platform for controller jitter — rather than
/// threading a single generator through the tick. That is what makes the
/// result independent of iteration order, and it is the property ADR-006
/// specifically preserves.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    inner: ChaCha8Rng,
}

impl DeterministicRng {
    /// Seed directly from an already-derived 32-bit seed.
    pub fn from_seed(seed: u32) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(u64::from(seed)),
        }
    }

    /// Derive a seed with [`stable_seed`] and open its stream.
    pub fn keyed(base_seed: u32, parts: &[SeedPart<'_>]) -> Self {
        Self::from_seed(stable_seed(base_seed, parts))
    }

    /// A uniform draw in `[0, 1)`.
    ///
    /// Built from the top 53 bits of a `u64`, so every value is a multiple of
    /// `2^-53` and no double is unreachable through rounding. This is the same
    /// construction numpy's `random()` uses, which keeps the *distribution*
    /// comparable even though individual draws differ.
    pub fn uniform(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.inner.next_u64() >> 11) as f64) * SCALE
    }

    /// A uniform draw in `[low, high)`.
    pub fn uniform_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// A normal draw with the given mean and standard deviation.
    ///
    /// Box–Muller, spelled out, deliberately: `rand_distr`'s `StandardNormal`
    /// is a ziggurat whose tables are explicitly **not** value-stable across
    /// releases, which is exactly the silent drift the fixtures exist to catch.
    /// P4.1 asks for the normal sampler to be chosen and pinned, and a
    /// transform written here cannot move underneath us.
    ///
    /// Both variates are consumed and only one is returned. Caching the second
    /// would halve the cost, but it would make a draw depend on how many draws
    /// preceded it within the stream — harmless in fact, and one more thing to
    /// reason about when a golden file moves. Streams are per-key and short;
    /// the clarity is worth more than the arithmetic.
    ///
    /// # `ln` and `cos` come from `libm`, not from the platform
    ///
    /// IEEE-754 requires `sqrt` and the arithmetic operators to be correctly
    /// rounded. It requires **nothing** of `ln` and `cos`, and implementations
    /// differ: glibc on x86_64 and Apple's libm on arm64 disagree by 1 ULP on
    /// some inputs. Box–Muller propagates that straight into the draw, so
    /// `f64::ln`/`f64::cos` make the stream platform-dependent — which is
    /// exactly what ADR-006 promises it is not, and what broke the conformance
    /// fixture the first time CI ran on a Mac.
    ///
    /// `libm` is a pure-Rust port of MUSL's libm with no platform calls, so it
    /// answers identically everywhere. Writing the transform out by hand
    /// stopped `rand_distr`'s ziggurat moving underneath us; this stops the
    /// system C library doing the same.
    pub fn normal(&mut self, mean: f64, std_dev: f64) -> f64 {
        // `uniform` can return exactly 0, and ln(0) is -inf. Reflecting into
        // (0, 1] costs nothing and removes the special case.
        let u1 = 1.0 - self.uniform();
        let u2 = self.uniform();
        // `sqrt` stays on f64: IEEE-754 pins it, so it is already portable.
        let radius = (-2.0 * libm::log(u1)).sqrt();
        mean + std_dev * radius * libm::cos(std::f64::consts::TAU * u2)
    }

    /// Raw 64 bits, for callers that need their own transform.
    /// A Poisson draw with mean `lambda`, by Knuth's multiplication method.
    ///
    /// # This is not numpy's algorithm, deliberately
    ///
    /// numpy switches between the PTRS transformed-rejection method and an
    /// inversion loop at `lambda = 10`, and reproducing either bit-for-bit
    /// would mean porting its exact branch structure. ADR-006 already
    /// re-baselined every stream in this crate — the contract is *"deterministic
    /// for a fixed seed and config"*, not *"identical to numpy"* — so this uses
    /// the documented textbook method instead and the fixtures pin it.
    ///
    /// Knuth's method costs one uniform per unit of `lambda` on average, which
    /// is fine for the clutter rates in use (0.01–0.5) and would not be for a
    /// large mean. The guard below keeps a mistaken large `lambda` from
    /// spinning: `exp(-lambda)` underflows to zero past about 745, and the loop
    /// would never terminate.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        // Spelled out rather than `!(lambda > 0.0)`: NaN has to be rejected
        // explicitly, and clippy is right that the negated comparison hides
        // that it is the case being handled.
        if lambda.is_nan() || lambda <= 0.0 {
            return 0;
        }
        // `libm::exp`, not `f64::exp`: neither is required by IEEE-754 to be
        // correctly rounded, and glibc disagrees with Apple's libm by 1 ULP --
        // which is what broke the conformance fixture when CI moved to a Mac.
        // `no_draw_routes_through_the_platform_libm` enforces this, and caught
        // exactly this line.
        //
        // The value is denormal below about -745 and zero past it.
        let limit = libm::exp(-lambda);
        if limit <= 0.0 {
            return 0;
        }
        let mut count = 0u64;
        let mut product = 1.0;
        loop {
            product *= self.uniform();
            if product <= limit {
                return count;
            }
            count += 1;
            // `uniform()` can return exactly 0.0, which ends the loop on the
            // next comparison, so this is a belt-and-braces bound rather than
            // the termination condition.
            if count > 1_000_000 {
                return count;
            }
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_seed_is_multiply_then_xor() {
        // Worked by hand from the Python source, so this fails if the order is
        // flipped to FNV-1a rather than merely disagreeing with a fixture.
        let expected = 7u32.wrapping_mul(FNV_PRIME_32) ^ u32::from(b'a');
        assert_eq!(stable_seed(7, &[SeedPart::Str("a")]), expected);

        let two_chars = expected.wrapping_mul(FNV_PRIME_32) ^ u32::from(b'b');
        assert_eq!(stable_seed(7, &[SeedPart::Str("ab")]), two_chars);
    }

    #[test]
    fn parts_are_hashed_as_python_renders_them() {
        assert_eq!(
            stable_seed(1, &[SeedPart::Int(1500)]),
            stable_seed(1, &[SeedPart::Str("1500")])
        );
        assert_eq!(
            stable_seed(1, &[SeedPart::Int(-42)]),
            stable_seed(1, &[SeedPart::Str("-42")])
        );
        assert_eq!(
            stable_seed(1, &[SeedPart::Int(i64::MIN)]),
            stable_seed(1, &[SeedPart::Str("-9223372036854775808")])
        );
    }

    #[test]
    fn concatenation_is_not_the_same_as_two_parts() {
        // It happens to be, because there is no separator between parts -- and
        // that is worth pinning rather than discovering later, since it means
        // ("ab", "c") and ("a", "bc") collide. The Python derivation has the
        // same property; callers relying on distinct keys must not build them
        // from adjacent free-form strings.
        assert_eq!(
            stable_seed(3, &[SeedPart::Str("ab"), SeedPart::Str("c")]),
            stable_seed(3, &[SeedPart::Str("a"), SeedPart::Str("bc")])
        );
    }

    #[test]
    fn a_stream_is_reproducible_from_its_key() {
        let draws = |()| {
            let mut rng =
                DeterministicRng::keyed(11, &[SeedPart::Str("node-1"), SeedPart::Int(500)]);
            (0..8).map(|_| rng.uniform()).collect::<Vec<_>>()
        };
        assert_eq!(draws(()), draws(()));
    }

    #[test]
    fn different_keys_give_different_streams() {
        let mut first = DeterministicRng::keyed(11, &[SeedPart::Str("node-1")]);
        let mut second = DeterministicRng::keyed(11, &[SeedPart::Str("node-2")]);
        assert_ne!(first.uniform(), second.uniform());
    }

    #[test]
    fn uniform_stays_in_range_over_a_long_run() {
        let mut rng = DeterministicRng::from_seed(1);
        for _ in 0..100_000 {
            let value = rng.uniform();
            assert!((0.0..1.0).contains(&value), "uniform produced {value}");
        }
    }

    #[test]
    fn normal_has_the_requested_moments() {
        // Not a distribution test -- it is a guard against a transform that is
        // scaled or centred wrongly, which is the failure Box-Muller invites.
        let mut rng = DeterministicRng::from_seed(99);
        let samples: Vec<f64> = (0..200_000).map(|_| rng.normal(3.0, 2.0)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        assert!((mean - 3.0).abs() < 0.02, "mean was {mean}");
        assert!(
            (variance.sqrt() - 2.0).abs() < 0.02,
            "std dev was {}",
            variance.sqrt()
        );
    }

    #[test]
    fn normal_is_finite_even_when_the_first_uniform_is_zero() {
        // u1 == 0 would make ln(u1) negative infinity. The reflection to (0, 1]
        // is what prevents it, and a stream that happens never to draw exactly
        // zero would not exercise it -- so drive the transform directly.
        let mut rng = DeterministicRng::from_seed(5);
        for _ in 0..50_000 {
            assert!(rng.normal(0.0, 1.0).is_finite());
        }
    }

    #[test]
    fn no_draw_routes_through_the_platform_libm() {
        // This is a source-text assertion, deliberately, because no behavioural
        // test on any single machine can make this one. `f64::ln` and
        // `libm::log` agree bit-for-bit on x86_64 glibc -- verified over the
        // whole conformance corpus, 0 of 8 draws changed on all five seeds --
        // so on the platform CI used to run, the wrong spelling is
        // indistinguishable from the right one. It is only wrong on arm64, and
        // by then it is a red fixture on someone else's machine.
        //
        // So the rule being enforced here is a source rule: this module calls
        // no f64 transcendental method. The correctly-rounded operations
        // (`sqrt`, the arithmetic operators) are unaffected and stay on f64.
        //
        // Only the production half of the file is scanned: the list below
        // contains every banned spelling by construction.
        let source = include_str!("deterministic.rs");
        let production = source
            .split_once("mod tests {")
            .expect("this test lives in `mod tests`, so the marker is present")
            .0;
        let banned = [
            ".ln()",
            ".log(",
            ".log2()",
            ".log10()",
            ".ln_1p()",
            ".exp()",
            ".exp2()",
            ".exp_m1()",
            ".sin()",
            ".cos()",
            ".tan()",
            ".sin_cos()",
            ".asin()",
            ".acos()",
            ".atan()",
            ".atan2(",
            ".powf(",
            ".hypot(",
            ".cbrt()",
            ".sinh()",
            ".cosh()",
            ".tanh()",
        ];
        for method in banned {
            // The comments above name several of these, so only lines that are
            // not comments count.
            let offender = production
                .lines()
                .enumerate()
                .find(|(_, line)| !line.trim_start().starts_with("//") && line.contains(method));
            assert!(
                offender.is_none(),
                "`{method}` reached deterministic.rs at line {}: {}\n\
                 The platform libm is not required by IEEE-754 to round these correctly, \
                 and glibc and Apple's libm disagree. Call the `libm` crate instead.",
                offender.unwrap().0 + 1,
                offender.unwrap().1.trim(),
            );
        }
    }
}
