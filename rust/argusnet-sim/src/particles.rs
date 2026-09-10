//! Particle offsets for the grid localizer, drawn per (platform, step).
//!
//! P4.2a ported every localizer formula against *recorded* offsets, leaving the
//! draw itself to this half. Under ADR-006 the draw cannot be bit-compatible
//! with Python — numpy's PCG64 and `SeedSequence` are not reproducible here —
//! so the contract is the one ADR-006 sets: deterministic for a fixed seed and
//! config, validated against declared statistical envelopes rather than a
//! golden vector.
//!
//! # Two behaviours that deliberately differ from Python
//!
//! `GridLocalizer` in Python draws from **one** generator, `default_rng(42)`,
//! created once and consumed sequentially by every platform on every step. Two
//! properties follow from that, and both were measured on the reference before
//! this was written:
//!
//! 1. **It is not order-independent.** Visiting `[drone-1, drone-2]` and
//!    `[drone-2, drone-1]` swaps the two platforms' estimates outright, because
//!    each takes whichever block of the shared stream its turn lands on. The
//!    rest of the simulation does not behave this way: `build_observations`
//!    keys a fresh generator per `(node, target, tick)` precisely so that
//!    reordering cannot change a result, and ADR-006 names preserving that
//!    keying as a requirement.
//!
//! 2. **`reset` does not restore the draw sequence.** It clears estimates, step
//!    counts and timeouts but never touches `_rng`, so a reset localizer
//!    produces different numbers from a fresh one — measured at ~2 m of
//!    position estimate on the first step after a reset. P4.1 defines reset as
//!    returning the runtime to a reproducible state; the reference does not.
//!
//! This sampler keys per `(platform, step)` instead, which gives both
//! properties by construction: the draw for a platform depends on nothing but
//! its own identity, its own step count and the run seed. It is a behaviour
//! change, taken deliberately rather than inherited, and it is the change
//! ADR-006's envelopes exist to license. See
//! `docs/migration/LOCALIZER_PRNG.md`.

use crate::deterministic::{DeterministicRng, SeedPart};

/// Namespace for this sampler's keys, so its stream cannot collide with another
/// subsystem that happens to key on the same platform and tick.
const STREAM: &str = "localizer.particles";

/// Draw `count` particle offsets around the origin, each coordinate normal with
/// standard deviation `scale`.
///
/// The result is usable directly as [`argusnet_localization::ParticleOffsets`].
/// `scale` comes from `GridLocalizer::particle_scale`, which owns the annealing
/// and coverage shrinkage — this function only draws.
///
/// Keyed on `(platform_id, step)`: the same arguments always give the same
/// offsets, and no other platform's draws can perturb them.
pub fn particle_offsets(
    base_seed: u32,
    platform_id: &str,
    step: u32,
    count: usize,
    scale: f64,
) -> Vec<[f64; 2]> {
    let mut rng = offset_stream(base_seed, platform_id, step);
    (0..count)
        .map(|_| [rng.normal(0.0, scale), rng.normal(0.0, scale)])
        .collect()
}

/// The keyed stream `particle_offsets` draws from.
///
/// Exposed so a caller that needs the offsets in a pre-allocated buffer can
/// drive the same stream without going through `Vec`, and so tests can assert
/// the keying directly rather than inferring it from outputs.
pub fn offset_stream(base_seed: u32, platform_id: &str, step: u32) -> DeterministicRng {
    DeterministicRng::keyed(
        base_seed,
        &[
            SeedPart::Str(STREAM),
            SeedPart::Str(platform_id),
            SeedPart::Int(i64::from(step)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Declared envelopes. These are derived from the sampling distribution,
    /// not fitted to what this code happens to emit -- a tolerance read off the
    /// output would pass for any implementation, including a broken one.
    ///
    /// For `n` draws from `N(0, s)`, the sample mean has standard error
    /// `s/sqrt(n)` and the sample standard deviation has standard error
    /// roughly `s/sqrt(2n)`. The bounds below are five of each, so a correct
    /// sampler fails about once in 3.5 million runs.
    ///
    /// `n` is what sets the *detection floor*, and it was chosen by measuring
    /// it rather than by picking a round number: at `n = 20_000` the mean bound
    /// is `0.035 s`, and a sampler biased by `0.02 s` slipped through the test
    /// unnoticed. At `n = 200_000` the bound is `0.0112 s` for the mean and
    /// `0.0079 s` for the spread, which catches that bias and any spread error
    /// above about 1%.
    const N: usize = 200_000;
    /// Standard errors, not magic numbers: `s/sqrt(n)` for the mean and
    /// `s/sqrt(2n)` for the standard deviation, each times five. Written as
    /// functions only because `sqrt` is not available in a `const`.
    fn mean_tolerance() -> f64 {
        5.0 / (N as f64).sqrt()
    }
    fn std_tolerance() -> f64 {
        5.0 / (2.0 * N as f64).sqrt()
    }

    fn moments(values: &[f64]) -> (f64, f64) {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, variance.sqrt())
    }

    fn flatten(offsets: &[[f64; 2]]) -> Vec<f64> {
        offsets.iter().flat_map(|o| [o[0], o[1]]).collect()
    }

    #[test]
    fn draws_land_within_the_declared_envelope() {
        for scale in [1.0, 80.0, 0.05] {
            let values = flatten(&particle_offsets(1234, "drone-1", 1, N, scale));
            let (mean, std) = moments(&values);
            assert!(
                mean.abs() <= mean_tolerance() * scale,
                "scale {scale}: mean {mean} outside +/-{} (5 standard errors)",
                mean_tolerance() * scale
            );
            assert!(
                (std - scale).abs() <= std_tolerance() * scale,
                "scale {scale}: std {std} not within {} of {scale}",
                std_tolerance() * scale
            );
        }
    }

    /// Box-Muller pairs a radius with an angle, so a bug that reused one
    /// variate would leave the two coordinates correlated while each still
    /// passed the moment checks above.
    #[test]
    fn the_two_coordinates_are_uncorrelated() {
        let offsets = particle_offsets(1234, "drone-1", 1, N, 80.0);
        let xs: Vec<f64> = offsets.iter().map(|o| o[0]).collect();
        let ys: Vec<f64> = offsets.iter().map(|o| o[1]).collect();
        let (mean_x, std_x) = moments(&xs);
        let (mean_y, std_y) = moments(&ys);
        let covariance = xs
            .iter()
            .zip(&ys)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / (xs.len() as f64 - 1.0);
        let correlation = covariance / (std_x * std_y);
        // Standard error of a correlation at n draws is about 1/sqrt(n).
        let tolerance = 5.0 / (N as f64).sqrt();
        assert!(
            correlation.abs() <= tolerance,
            "coordinates correlated at {correlation}, beyond +/-{tolerance}"
        );
    }

    #[test]
    fn the_same_key_always_gives_the_same_offsets() {
        let first = particle_offsets(1234, "drone-1", 7, 32, 80.0);
        let second = particle_offsets(1234, "drone-1", 7, 32, 80.0);
        assert_eq!(first, second);
    }

    /// The property Python's shared stream does not have. Drawing for one
    /// platform cannot move another's offsets, so the order platforms are
    /// visited in is irrelevant -- and a reset, which only clears step counts,
    /// returns the very same draws.
    #[test]
    fn a_platforms_offsets_do_not_depend_on_who_else_drew() {
        let alone = particle_offsets(1234, "drone-2", 3, 16, 80.0);
        // Interleave other platforms and other steps around it.
        let _ = particle_offsets(1234, "drone-1", 3, 16, 80.0);
        let _ = particle_offsets(1234, "drone-3", 1, 16, 80.0);
        let _ = particle_offsets(1234, "drone-2", 2, 16, 80.0);
        let after = particle_offsets(1234, "drone-2", 3, 16, 80.0);
        assert_eq!(alone, after, "another platform's draw perturbed this one");
    }

    /// The namespace exists so another subsystem keyed on the same platform and
    /// tick cannot draw this stream's numbers. Without this, dropping it from
    /// the key changes every draw and no test notices.
    #[test]
    fn the_stream_namespace_is_part_of_the_key() {
        let mut ours = offset_stream(1234, "drone-1", 3);
        let mut unnamespaced =
            DeterministicRng::keyed(1234, &[SeedPart::Str("drone-1"), SeedPart::Int(3)]);
        assert_ne!(ours.next_u64(), unnamespaced.next_u64());

        let mut other_subsystem = DeterministicRng::keyed(
            1234,
            &[
                SeedPart::Str("some.other.stream"),
                SeedPart::Str("drone-1"),
                SeedPart::Int(3),
            ],
        );
        let mut ours_again = offset_stream(1234, "drone-1", 3);
        assert_ne!(ours_again.next_u64(), other_subsystem.next_u64());
    }

    #[test]
    fn different_keys_give_different_streams() {
        let base = particle_offsets(1234, "drone-1", 1, 16, 80.0);
        assert_ne!(base, particle_offsets(1234, "drone-2", 1, 16, 80.0));
        assert_ne!(base, particle_offsets(1234, "drone-1", 2, 16, 80.0));
        assert_ne!(base, particle_offsets(9999, "drone-1", 1, 16, 80.0));
    }

    /// Scale multiplies the standard normal, so the same key at two scales
    /// gives the same shape. Pinning the ratio catches a scale applied to only
    /// one coordinate, or applied twice.
    #[test]
    fn scale_is_a_pure_multiplier_on_a_fixed_key() {
        let unit = particle_offsets(1234, "drone-1", 1, 64, 1.0);
        let scaled = particle_offsets(1234, "drone-1", 1, 64, 12.5);
        for (u, s) in unit.iter().zip(&scaled) {
            for axis in 0..2 {
                assert!(
                    (s[axis] - u[axis] * 12.5).abs() <= 1e-12 * s[axis].abs().max(1.0),
                    "axis {axis}: {} is not {} * 12.5",
                    s[axis],
                    u[axis]
                );
            }
        }
    }

    #[test]
    fn degenerate_counts_and_scales_are_not_special_cases() {
        assert!(particle_offsets(1234, "drone-1", 1, 0, 80.0).is_empty());
        assert_eq!(
            particle_offsets(1234, "drone-1", 1, 4, 0.0),
            [[0.0, 0.0]; 4]
        );
    }
}
