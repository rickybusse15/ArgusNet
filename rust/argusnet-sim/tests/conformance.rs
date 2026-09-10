//! RNG and math conformance fixtures.
//!
//! P4.1 asks for these *before* any stochastic stage moves, and they do two
//! different jobs:
//!
//! - `seed_derivation_conformance.json` is generated from Python's own
//!   `_stable_seed`. ADR-006 re-baselines the random *stream* but explicitly
//!   preserves the seed-derivation *scheme*, so these values must match
//!   bit-for-bit across the port. A failure here means the port diverged from
//!   the reference, not that a baseline moved.
//!
//! - `rng_stream_conformance.json` is generated from this crate and records
//!   what the pinned stream actually produces. Nothing outside says these are
//!   the "right" numbers — the point is that they must not change *silently*.
//!   A dependency bump that alters a draw fails here instead of quietly moving
//!   every golden file in the repo.
//!
//! The second file is a baseline and may legitimately be regenerated; when it
//! is, that belongs in a commit that changes nothing else, per ADR-006's rule
//! for the re-baseline itself.

use argusnet_sim::deterministic::{stable_seed, DeterministicRng, SeedPart};
use serde::Deserialize;

/// Fixtures live at the repository root, and this crate sits two levels down.
fn fixture(name: &str) -> String {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("failed to read {path}: {error}"))
}

#[derive(Deserialize)]
struct SeedCases {
    cases: Vec<SeedCase>,
}

#[derive(Deserialize)]
struct SeedCase {
    base_seed: u32,
    parts: Vec<Part>,
    seed: u32,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum Part {
    Str { value: String },
    Int { value: i64 },
}

#[test]
fn seed_derivation_matches_the_python_reference() {
    let parsed: SeedCases = serde_json::from_str(&fixture("seed_derivation_conformance.json"))
        .expect("seed fixture is valid JSON");
    assert!(
        parsed.cases.len() >= 30,
        "fixture shrank to {} cases; regenerate it rather than weakening the check",
        parsed.cases.len()
    );

    for (index, case) in parsed.cases.iter().enumerate() {
        let parts: Vec<SeedPart<'_>> = case
            .parts
            .iter()
            .map(|part| match part {
                Part::Str { value } => SeedPart::Str(value.as_str()),
                Part::Int { value } => SeedPart::Int(*value),
            })
            .collect();
        let actual = stable_seed(case.base_seed, &parts);
        assert_eq!(
            actual,
            case.seed,
            "case {index}: base_seed={} parts={:?} -- Rust derived {actual}, Python derived {}",
            case.base_seed,
            case.parts
                .iter()
                .map(|part| match part {
                    Part::Str { value } => value.clone(),
                    Part::Int { value } => value.to_string(),
                })
                .collect::<Vec<_>>(),
            case.seed
        );
    }
}

#[derive(Deserialize)]
struct StreamCases {
    cases: Vec<StreamCase>,
}

/// Draws are compared as raw bit patterns, not as decimals.
///
/// The first version of this fixture stored decimals and failed on its own
/// output: `serde_json` round-tripped `0.9354265029131292` back as
/// `...291`. A determinism fixture that cannot reproduce its own values is
/// worse than none, so `*_bits` is the contract and the decimals beside it
/// are for a human reading the diff.
#[derive(Deserialize)]
struct StreamCase {
    seed: u32,
    uniform_bits: Vec<u64>,
    normal_bits: Vec<u64>,
}

#[test]
fn the_pinned_stream_still_produces_the_recorded_draws() {
    let parsed: StreamCases = serde_json::from_str(&fixture("rng_stream_conformance.json"))
        .expect("stream fixture is valid JSON");

    for case in &parsed.cases {
        let mut rng = DeterministicRng::from_seed(case.seed);
        let uniform: Vec<u64> = (0..case.uniform_bits.len())
            .map(|_| rng.uniform().to_bits())
            .collect();
        assert_eq!(
            uniform, case.uniform_bits,
            "uniform draws changed for seed {}. If a dependency was bumped, this is \
             the drift the pin exists to catch: re-baseline deliberately, in a commit \
             that changes nothing else.",
            case.seed
        );

        let mut rng = DeterministicRng::from_seed(case.seed);
        let normal: Vec<u64> = (0..case.normal_bits.len())
            .map(|_| rng.normal(0.0, 1.0).to_bits())
            .collect();
        assert_eq!(
            normal, case.normal_bits,
            "normal draws changed for seed {}. The transform is written out in \
             deterministic.rs precisely so this cannot move underneath us.",
            case.seed
        );
    }
}

/// The property that actually matters, asserted rather than assumed.
///
/// ADR-006 keeps the per-key stream scheme because it makes a tick's result
/// independent of the order platforms are visited in. That is worth a test:
/// a future refactor that threads one shared generator through the tick would
/// still pass every fixture above and silently break this.
#[test]
fn per_key_streams_make_the_result_independent_of_iteration_order() {
    let keys = ["node-0", "node-1", "node-2", "drone-alpha"];
    let draw_all = |order: &[usize]| {
        let mut results: Vec<(String, f64)> = order
            .iter()
            .map(|&index| {
                let key = keys[index];
                let mut rng =
                    DeterministicRng::keyed(1234, &[SeedPart::Str(key), SeedPart::Int(1500)]);
                (key.to_string(), rng.normal(0.0, 0.01))
            })
            .collect();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        results
    };

    assert_eq!(draw_all(&[0, 1, 2, 3]), draw_all(&[3, 1, 0, 2]));
    assert_eq!(draw_all(&[0, 1, 2, 3]), draw_all(&[2, 0, 3, 1]));
}

/// Regenerates the stream baseline. Ignored by default: it rewrites a committed
/// file, and doing that silently as part of a normal test run would defeat the
/// point of having a baseline at all.
///
/// ```text
/// cargo test -p argusnet-sim --test conformance -- --ignored regenerate
/// ```
#[test]
#[ignore = "rewrites a committed baseline; run deliberately"]
fn regenerate_stream_conformance_fixture() {
    let mut cases = Vec::new();
    for seed in [0u32, 1, 1234, 99, u32::MAX] {
        let mut rng = DeterministicRng::from_seed(seed);
        let uniform: Vec<f64> = (0..8).map(|_| rng.uniform()).collect();
        let mut rng = DeterministicRng::from_seed(seed);
        let normal: Vec<f64> = (0..8).map(|_| rng.normal(0.0, 1.0)).collect();
        cases.push(serde_json::json!({
            "seed": seed,
            "uniform_bits": uniform.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "normal_bits": normal.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            "uniform_decimal_for_humans": uniform,
            "normal_decimal_for_humans": normal,
        }));
    }
    let document = serde_json::json!({
        "source": "rust/argusnet-sim/src/deterministic.rs",
        "note": "A baseline, not a reference. These are what the pinned stream produces; \
                 nothing outside this repository says they are the right numbers. The point \
                 is that they must not change silently. Re-baseline deliberately, in a commit \
                 that changes nothing else (ADR-006). The *_bits fields are the contract: JSON \
                 decimals do not round-trip exactly, so the decimals beside them are for reading \
                 a diff, not for comparison.",
        "cases": cases,
    });
    let path = format!(
        "{}/../../tests/fixtures/rng_stream_conformance.json",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(
        &path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&document).expect("serializable")
        ),
    )
    .expect("fixture is writable");
    println!("wrote {path}");
}
