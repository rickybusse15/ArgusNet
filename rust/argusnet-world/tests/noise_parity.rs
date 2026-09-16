//! Parity for `procedural.py`'s deterministic primitives, bit-for-bit.
//!
//! These sit at the bottom of the terrain dependency graph, so every
//! comparison is on raw bit patterns and never inside a tolerance: a ULP here
//! is a divergence in every terrain the runtime builds.

use argusnet_world::noise::{
    arange, fbm, grid_values, hash_noise, line_gaussian, smoothstep, stable_seed, value_noise,
    Bounds2D, FbmConfig, LineEnvelope,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    stable_seeds: Vec<SeedCase>,
    smoothsteps: Vec<SmoothstepCase>,
    hash_noise: Vec<HashCase>,
    value_noise: Vec<NoiseCase>,
    fbm: Vec<FbmCase>,
    line_gaussian: Vec<LineCase>,
    grid_values: Vec<GridCase>,
}

#[derive(Deserialize)]
struct SeedCase {
    parts: Vec<String>,
    seed: String,
}

#[derive(Deserialize)]
struct SmoothstepCase {
    value: String,
    result: String,
}

#[derive(Deserialize)]
struct HashCase {
    ix: String,
    iy: String,
    seed: String,
    result: String,
}

#[derive(Deserialize)]
struct NoiseCase {
    x: String,
    y: String,
    seed: String,
    result: String,
}

#[derive(Deserialize)]
struct FbmCase {
    x: String,
    y: String,
    seed: String,
    base_wavelength_m: String,
    octaves: String,
    persistence: String,
    lacunarity: String,
    result: String,
}

#[derive(Deserialize)]
struct LineCase {
    xx: String,
    yy: String,
    angle_rad: String,
    offset_m: String,
    width_m: String,
    length_m: String,
    center_xy: [String; 2],
    result: String,
}

#[derive(Deserialize)]
struct GridCase {
    x_min_m: String,
    x_max_m: String,
    y_min_m: String,
    y_max_m: String,
    resolution_m: String,
    x_values: Vec<String>,
    y_values: Vec<String>,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn i(text: &str) -> i64 {
    text.parse().expect("fixture integer")
}

/// How far `line_gaussian` may sit from the reference's.
///
/// It is the one function in this fixture that routes through `sin`, `cos` and
/// `exp`, whose last bit is a property of the platform's libm rather than of
/// the port -- see `SLOPE_TOLERANCE_RAD` in `grid_ops_parity.rs` and ADR-006.
/// Measured against the `libm` crate, a third implementation: **one case of the
/// fixture moves, by exactly 1 ULP (1.1e-16)**. This bound is ~9,000x that,
/// while the result is a normalised envelope in [0, 1], so a port that squared
/// the along-track term instead of raising it to the fourth power -- the
/// mistake this fixture exists to catch -- misses by order 0.1.
///
/// Every other case here stays bit-exact, and was confirmed bit-identical
/// under the same probe.
const LINE_GAUSSIAN_TOLERANCE: f64 = 1.0e-12;

fn near(actual: f64, expected: &str, what: &str) {
    let want = f(expected);
    assert!(
        (actual - want).abs() <= LINE_GAUSSIAN_TOLERANCE,
        "{what}: got {actual}, expected {expected} (off by {:e}, tolerance {:e})",
        (actual - want).abs(),
        LINE_GAUSSIAN_TOLERANCE
    );
}

fn same(actual: f64, expected: &str, what: &str) {
    assert_eq!(
        actual.to_bits(),
        f(expected).to_bits(),
        "{what}: got {actual}, expected {expected}"
    );
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/noise.json"
    );
    let text = std::fs::read_to_string(path).expect("noise fixture");
    serde_json::from_str(&text).expect("noise fixture parses")
}

#[test]
fn stable_seeds_match_the_reference() {
    let fixture = load();
    assert!(!fixture.stable_seeds.is_empty());
    for case in &fixture.stable_seeds {
        let parts: Vec<&str> = case.parts.iter().map(String::as_str).collect();
        assert_eq!(
            stable_seed(&parts),
            i(&case.seed),
            "stable_seed({:?})",
            case.parts
        );
    }
}

/// `procedural.py::_stable_seed` and `sim.py::_stable_seed` share a name and
/// are different functions. If a future "cleanup" unifies them, this fails
/// before any terrain silently changes.
#[test]
fn the_terrain_seed_is_not_the_fnv_walk_of_the_same_name() {
    let fixture = load();
    let fnv1 = |parts: &[String]| -> i64 {
        let mut seed: u32 = 0;
        for part in parts {
            for character in part.chars() {
                seed = seed.wrapping_mul(16_777_619) ^ (character as u32);
            }
        }
        seed as i64
    };
    let distinguishing = fixture
        .stable_seeds
        .iter()
        .filter(|case| i(&case.seed) != fnv1(&case.parts))
        .count();
    assert!(
        distinguishing > 0,
        "every case agrees with the FNV-1 walk; the two same-named seed \
         functions are not being distinguished"
    );
}

#[test]
fn smoothstep_matches_the_reference() {
    let fixture = load();
    for case in &fixture.smoothsteps {
        same(
            smoothstep(f(&case.value)),
            &case.result,
            &format!("smoothstep({})", case.value),
        );
    }
}

#[test]
fn hash_noise_matches_the_reference() {
    let fixture = load();
    for case in &fixture.hash_noise {
        same(
            hash_noise(i(&case.ix), i(&case.iy), i(&case.seed)),
            &case.result,
            &format!("hash_noise({}, {}, {})", case.ix, case.iy, case.seed),
        );
    }
}

/// The multiply chain must actually exceed `i64` range somewhere, or a port
/// using a plain `*` would pass — and then panic in a debug build in
/// production rather than in the test suite.
#[test]
fn the_hash_fixture_still_overflows_i64() {
    let fixture = load();
    let overflowing = fixture
        .hash_noise
        .iter()
        .filter(|case| {
            let h = i(&case.ix).wrapping_mul(73_856_093)
                ^ i(&case.iy).wrapping_mul(19_349_663)
                ^ i(&case.seed);
            ((h >> 13) ^ h).checked_mul(0x045D_9F3B).is_none()
        })
        .count();
    assert!(
        overflowing > 0,
        "no hash_noise case overflows i64; the wrapping multiply is untested"
    );
}

#[test]
fn value_noise_matches_the_reference() {
    let fixture = load();
    for case in &fixture.value_noise {
        same(
            value_noise(f(&case.x), f(&case.y), i(&case.seed)),
            &case.result,
            &format!("value_noise({}, {}, {})", case.x, case.y, case.seed),
        );
    }
}

#[test]
fn fbm_matches_the_reference() {
    let fixture = load();
    for case in &fixture.fbm {
        let config = FbmConfig {
            seed: i(&case.seed),
            base_wavelength_m: f(&case.base_wavelength_m),
            octaves: i(&case.octaves),
            persistence: f(&case.persistence),
            lacunarity: f(&case.lacunarity),
        };
        same(
            fbm(f(&case.x), f(&case.y), config),
            &case.result,
            &format!(
                "fbm({}, {}, seed={}, wl={}, oct={}, p={}, l={})",
                case.x,
                case.y,
                case.seed,
                case.base_wavelength_m,
                case.octaves,
                case.persistence,
                case.lacunarity
            ),
        );
    }
}

#[test]
fn line_gaussian_matches_the_reference() {
    let fixture = load();
    for case in &fixture.line_gaussian {
        let envelope = LineEnvelope {
            angle_rad: f(&case.angle_rad),
            offset_m: f(&case.offset_m),
            width_m: f(&case.width_m),
            length_m: f(&case.length_m),
            center_xy: (f(&case.center_xy[0]), f(&case.center_xy[1])),
        };
        near(
            line_gaussian(f(&case.xx), f(&case.yy), envelope),
            &case.result,
            &format!("line_gaussian({}, {})", case.xx, case.yy),
        );
    }
}

#[test]
fn grid_axes_match_the_reference() {
    let fixture = load();
    for case in &fixture.grid_values {
        let bounds = Bounds2D {
            x_min_m: f(&case.x_min_m),
            x_max_m: f(&case.x_max_m),
            y_min_m: f(&case.y_min_m),
            y_max_m: f(&case.y_max_m),
        };
        let (xs, ys) = grid_values(bounds, f(&case.resolution_m));
        let label = format!(
            "grid [{}..{}]x[{}..{}] @ {}",
            case.x_min_m, case.x_max_m, case.y_min_m, case.y_max_m, case.resolution_m
        );
        assert_eq!(xs.len(), case.x_values.len(), "{label}: x length");
        assert_eq!(ys.len(), case.y_values.len(), "{label}: y length");
        for (index, (actual, expected)) in xs.iter().zip(&case.x_values).enumerate() {
            same(*actual, expected, &format!("{label}: x[{index}]"));
        }
        for (index, (actual, expected)) in ys.iter().zip(&case.y_values).enumerate() {
            same(*actual, expected, &format!("{label}: y[{index}]"));
        }
    }
}

/// `np.arange` is not `start + i * step`, and it is not cumulative addition
/// either. Both wrong spellings must disagree with the fixture somewhere, or
/// the recomputed-delta rule is untested.
#[test]
fn the_grid_fixture_still_discriminates_the_arange_spellings() {
    let fixture = load();
    let mut caught_multiply = 0usize;
    let mut caught_cumulative = 0usize;

    for case in &fixture.grid_values {
        let step = f(&case.resolution_m);
        for (values, lo_text) in [
            (&case.x_values, &case.x_min_m),
            (&case.y_values, &case.y_min_m),
        ] {
            if values.len() < 3 {
                continue;
            }
            let lo = f(lo_text);
            if values
                .iter()
                .enumerate()
                .any(|(index, v)| f(v) != lo + index as f64 * step)
            {
                caught_multiply += 1;
            }
            let mut accumulated = lo;
            let mut differs = false;
            for v in values {
                if f(v) != accumulated {
                    differs = true;
                    break;
                }
                accumulated += step;
            }
            if differs {
                caught_cumulative += 1;
            }
        }
    }

    assert!(
        caught_multiply > 0,
        "no axis disagrees with `start + i * step`; the recomputed delta is untested"
    );
    assert!(
        caught_cumulative > 0,
        "no axis disagrees with cumulative addition; the recomputed delta is untested"
    );
}

/// The two lerp spellings — `a + t*(b - a)` and `(1 - t)*a + t*b` — must give
/// a different *final* `value_noise` somewhere.
///
/// Measuring the inner `nx0` instead is the trap: the outer lerp absorbs a
/// last-bit difference there almost every time. An earlier version of these
/// fixture cases was searched that way and pinned nothing. The rate on the
/// final output is about 2e-6 per random point, so these cases cannot be
/// stumbled into — if they are ever dropped, this fails.
#[test]
fn the_noise_fixture_still_pins_the_interpolation_form() {
    let fixture = load();
    let discriminating = fixture
        .value_noise
        .iter()
        .filter(|case| {
            let (x, y, seed) = (f(&case.x), f(&case.y), i(&case.seed));
            let ix = x.floor() as i64;
            let iy = y.floor() as i64;
            let fx = smoothstep(x - ix as f64);
            let fy = smoothstep(y - iy as f64);
            let n00 = hash_noise(ix, iy, seed);
            let n10 = hash_noise(ix + 1, iy, seed);
            let n01 = hash_noise(ix, iy + 1, seed);
            let n11 = hash_noise(ix + 1, iy + 1, seed);
            let nx1 = n01 + fx * (n11 - n01);

            let reference_nx0 = n00 + fx * (n10 - n00);
            let reference = reference_nx0 + fy * (nx1 - reference_nx0);
            let alternate_nx0 = (1.0 - fx) * n00 + fx * n10;
            let alternate = alternate_nx0 + fy * (nx1 - alternate_nx0);
            reference != alternate
        })
        .count();
    assert!(
        discriminating > 0,
        "no value_noise case distinguishes `a + t*(b-a)` from `(1-t)*a + t*b` \
         in the final result; the interpolation form is unpinned"
    );
}

/// The bare `arange` helper, on the case from the module docs.
#[test]
fn arange_matches_numpys_recomputed_delta() {
    let values = arange(0.1, 0.1 + 5.0 + 0.15, 0.3);
    let delta = values[1] - values[0];
    assert_ne!(delta, 0.3);
    for (index, value) in values.iter().enumerate().skip(2) {
        assert_eq!(*value, 0.1 + index as f64 * delta);
    }
}
