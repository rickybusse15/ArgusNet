//! Parity for numpy's `default_rng(seed).uniform(low, high)`.
//!
//! The fixture is captured from numpy itself, so this compares the port against
//! the reference rather than against a second statement of the same reasoning.
//!
//! Raw 64-bit outputs are compared **as well as** the scaled doubles, so a
//! failure says which layer broke: the entropy mixing, the bit generator, the
//! 53-bit double construction, or the affine scale.

use argusnet_world::numpy_rng::NumpyRng;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    raw: Vec<RawCase>,
    uniform: Vec<UniformCase>,
}

#[derive(Deserialize)]
struct RawCase {
    name: String,
    seed: u32,
    raw_u64: Vec<String>,
}

#[derive(Deserialize)]
struct UniformCase {
    name: String,
    seed: u32,
    draws: Vec<Draw>,
}

#[derive(Deserialize)]
struct Draw {
    low: String,
    high: String,
    value: String,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn load() -> Fixture {
    let path = format!(
        "{}/../../tests/fixtures/numpy_rng.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).expect("numpy rng fixture parses")
}

#[test]
fn the_raw_bit_generator_stream_matches_numpy() {
    let fixture = load();
    assert!(fixture.raw.len() >= 19, "seed count");
    for case in &fixture.raw {
        let mut rng = NumpyRng::new(case.seed);
        for (index, expected) in case.raw_u64.iter().enumerate() {
            let want: u64 = expected.parse().expect("fixture u64");
            let got = rng.next_u64();
            assert_eq!(
                got, want,
                "{} (seed {}): raw output {index} differs",
                case.name, case.seed
            );
        }
    }
}

#[test]
fn every_uniform_draw_matches_numpy_bit_for_bit() {
    let fixture = load();
    for case in &fixture.uniform {
        let mut rng = NumpyRng::new(case.seed);
        for (index, draw) in case.draws.iter().enumerate() {
            let got = rng.uniform(f(&draw.low), f(&draw.high));
            let want = f(&draw.value);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{} (seed {}): draw {index} over [{}, {}]: got {got}, expected {want}",
                case.name,
                case.seed,
                draw.low,
                draw.high
            );
        }
    }
}

#[test]
fn the_preset_seeds_are_covered() {
    // The adversarial seeds (0, 1, 2**31-1) are the ones most likely to expose
    // a mixing bug, but they are not what production uses. If the preset cases
    // ever fall out of the fixture, the port would be proven only on seeds
    // nothing draws from.
    let fixture = load();
    let presets = fixture
        .uniform
        .iter()
        .filter(|c| c.name.starts_with("preset/"))
        .count();
    assert!(
        presets >= 12,
        "only {presets} preset seeds in the fixture; every preset should be there"
    );
}

// ---------------------------------------------------------------------------
// Discrimination guards. Each implements a wrong version and requires the
// recorded stream to rule it out.
// ---------------------------------------------------------------------------

#[test]
fn stepping_after_the_read_instead_of_before_changes_the_stream() {
    // `pcg64_random_r` steps and THEN outputs. Reading first and stepping after
    // is the natural way to write it and yields a stream shifted by one, which
    // is uniform, deterministic and wrong.
    let fixture = load();
    let case = &fixture.raw[0];
    let mut rng = NumpyRng::new(case.seed);
    let correct: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    // The shifted stream: drop the first output and compare what remains.
    let mut shifted = NumpyRng::new(case.seed);
    shifted.next_u64();
    let off_by_one: Vec<u64> = (0..7).map(|_| shifted.next_u64()).collect();
    assert_ne!(
        correct[..7],
        off_by_one[..],
        "the stream is invariant to a one-step shift, so the step order is untested"
    );
}

#[test]
fn the_double_uses_the_top_53_bits_not_the_bottom() {
    // `(raw >> 11) * 2^-53`. Masking the low 53 bits instead gives a value in
    // the same range from the same stream. The two must disagree.
    let fixture = load();
    let mut disagreed = 0;
    for case in fixture.raw.iter().take(5) {
        let mut rng = NumpyRng::new(case.seed);
        for _ in 0..16 {
            let raw = rng.next_u64();
            let top = (raw >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0);
            let bottom = (raw & ((1u64 << 53) - 1)) as f64 * (1.0 / 9_007_199_254_740_992.0);
            if top.to_bits() != bottom.to_bits() {
                disagreed += 1;
            }
        }
    }
    assert!(
        disagreed > 70,
        "only {disagreed} of 80 raw outputs distinguish the top 53 bits from the bottom"
    );
}

#[test]
fn uniform_is_an_affine_scale_not_a_lerp() {
    // numpy computes `low + (high - low) * t`. The algebraically equal
    // `low * (1 - t) + high * t` rounds differently, and the fixture must be
    // able to tell them apart or the spelling is unpinned.
    let fixture = load();
    let mut disagreed = 0;
    let mut total = 0;
    for case in &fixture.uniform {
        let mut rng = NumpyRng::new(case.seed);
        for draw in &case.draws {
            let (low, high) = (f(&draw.low), f(&draw.high));
            let t = rng.next_f64();
            let affine = low + (high - low) * t;
            let lerp = low * (1.0 - t) + high * t;
            total += 1;
            if affine.to_bits() != lerp.to_bits() {
                disagreed += 1;
            }
        }
    }
    assert!(
        disagreed > 0,
        "none of {total} draws distinguishes the affine form from a lerp"
    );
}

#[test]
fn a_zero_width_range_returns_its_bound_and_still_consumes_a_draw() {
    // `low + 0.0 * t` is `low` for every finite t, so the value proves nothing
    // on its own -- what matters is that the draw was still taken. A port that
    // short-circuits on `low == high` would desynchronise everything after it.
    let mut rng = NumpyRng::new(7);
    let value = rng.uniform(5.0, 5.0);
    assert_eq!(value, 5.0);
    let after_consuming = rng.uniform(0.0, 1.0);

    let mut skipping = NumpyRng::new(7);
    let after_skipping = skipping.uniform(0.0, 1.0);
    assert_ne!(
        after_consuming.to_bits(),
        after_skipping.to_bits(),
        "the zero-width draw did not advance the stream"
    );
}

#[test]
fn numpy_rejects_a_reversed_range_so_the_port_is_only_defined_for_low_le_high() {
    // Recorded as a fact, not an assertion about this code: numpy raises
    // "ValueError: high - low < 0" rather than sorting or scaling by a
    // negative. There is therefore no reference behaviour to match, and every
    // caller in `procedural_terrain_grid` passes low <= high. This test exists
    // so that a future reader does not mistake the absence of a reversed case
    // in the fixture for an oversight.
    let mut rng = NumpyRng::new(1);
    let value = rng.uniform(-1.0, 1.0);
    assert!((-1.0..1.0).contains(&value));
}

/// `integers` interleaved with double draws, as the target behaviours and
/// mission zones draw them (P4.9b.2). `integers` reads 32 bits at a time
/// through PCG64's half-word buffer, which the double draws neither read nor
/// clear, so the operation order is what is under test -- including Lemire's
/// rejection loop (a range of 2^31 + 1) and the exact 2^32 - 1 path. The
/// doubles are `random()`, which is integer-exact on every platform; see the
/// capture script for why not `uniform()`.
#[test]
fn integers_interleaved_with_doubles_match_numpy() {
    #[derive(Deserialize)]
    struct Mixed {
        mixed: Vec<MixedCase>,
    }
    #[derive(Deserialize)]
    struct MixedCase {
        name: String,
        seed: u32,
        draws: Vec<MixedDraw>,
    }
    #[derive(Deserialize)]
    struct MixedDraw {
        op: String,
        low: serde_json::Value,
        high: serde_json::Value,
        value: String,
    }
    let path = format!(
        "{}/../../tests/fixtures/numpy_rng.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture: Mixed = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert!(fixture.mixed.len() >= 19, "seed count");
    let mut integers = 0;
    for case in &fixture.mixed {
        let mut rng = NumpyRng::new(case.seed);
        for (index, draw) in case.draws.iter().enumerate() {
            match draw.op.as_str() {
                "integers" => {
                    let got = rng.integers(draw.low.as_i64().unwrap(), draw.high.as_i64().unwrap());
                    assert_eq!(got.to_string(), draw.value, "{} draw {index}", case.name);
                    integers += 1;
                }
                "random" => {
                    assert_eq!(
                        rng.random().to_bits(),
                        f(&draw.value).to_bits(),
                        "{} draw {index}",
                        case.name
                    );
                }
                other => panic!("unknown op {other}"),
            }
        }
    }
    assert!(integers >= 684, "integer draws compared: {integers}");
}
