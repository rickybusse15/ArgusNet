//! numpy's `default_rng` stream, reproduced exactly.
//!
//! # Why this exists at all, against ADR-006's advice
//!
//! ADR-006 recommends **not** reproducing numpy's PRNG bit-for-bit, and
//! re-baselining on statistical envelopes instead. Its stated reason is cost:
//! *"possible but fragile and costly ... a hand-verified reimplementation of
//! two algorithms"*, and it was written about `build_observations`, which
//! constructs a **fresh generator per (node, target) pair per tick** and uses
//! `SeedSequence.spawn`.
//!
//! `procedural_terrain_grid` is not that. It creates **one** generator and
//! draws **12 to 46 scalars** from it — `rng.uniform(low, high)` and nothing
//! else. No `spawn`, no arrays, no other distribution. The cost premise behind
//! the recommendation does not hold, and the benefit is large: terrain built by
//! Rust is then *identical* to terrain baked by Python, so the scene packages
//! Python still writes during the transition describe the same world the Rust
//! runtime generates. A statistical envelope cannot give that.
//!
//! See `docs/adr/006-determinism-rebaseline.md`, which records this as a scoped
//! amendment rather than a departure.
//!
//! # What is reproduced, and what is deliberately not
//!
//! Only the path `default_rng(int)` → `Generator.uniform(low, high)` takes:
//! `SeedSequence` entropy mixing, `PCG64` seeding, the XSL-RR output, the
//! 53-bit double construction, and the affine scale. Nothing else in numpy's
//! RNG surface is implemented, because nothing else is used — `spawn`,
//! `integers`, `normal` and the array paths would each need their own proof and
//! none of them would be exercised.

/// numpy's `SeedSequence`, restricted to a single small integer of entropy.
///
/// The constants and the mixing order are numpy's
/// (`numpy/random/bit_generator.pyx`); they are a fixed part of the stream
/// definition, not tunable parameters.
mod seed_sequence {
    const INIT_A: u32 = 0x43B0_D7E5;
    const MULT_A: u32 = 0x931E_8875;
    const INIT_B: u32 = 0x8B51_F9DD;
    const MULT_B: u32 = 0x58F3_8DED;
    const MIX_L: u32 = 0xCA01_F9DD;
    const MIX_R: u32 = 0x4973_F715;
    /// numpy uses `32 // 2`. Every shift below is this one.
    const XSHIFT: u32 = 16;
    const POOL_SIZE: usize = 4;

    /// The entropy pool. Every multiply and subtract wraps: numpy is working in
    /// `uint32`, where overflow is defined, and Rust's `*` would panic.
    // The pair loops below index `mixer` by design: `hash` is called once per
    // (src, dst) PAIR and advances `hash_const` each time, and `mixer[src]` is
    // re-read after earlier destinations have already changed it. Neither loop
    // can become an iterator without changing the stream, so the indices are
    // the algorithm rather than a style choice.
    #[allow(clippy::needless_range_loop)]
    pub fn mix_entropy(entropy: &[u32]) -> [u32; POOL_SIZE] {
        let mut hash_const = INIT_A;
        let hash = |value: u32, hash_const: &mut u32| {
            let mut v = value ^ *hash_const;
            *hash_const = hash_const.wrapping_mul(MULT_A);
            v = v.wrapping_mul(*hash_const);
            v ^ (v >> XSHIFT)
        };
        let mix = |x: u32, y: u32| {
            let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y));
            r ^ (r >> XSHIFT)
        };

        let mut mixer = [0u32; POOL_SIZE];
        for (i, slot) in mixer.iter_mut().enumerate() {
            *slot = hash(entropy.get(i).copied().unwrap_or(0), &mut hash_const);
        }
        // Every ordered pair, skipping the diagonal. The nesting order matters:
        // `mixer[i_src]` is re-read after earlier destinations have changed it,
        // and `hash` is called once per PAIR, advancing `hash_const` each time,
        // so neither loop can be hoisted or turned into an iterator without
        // changing the stream. The indices are the algorithm.
        for src in 0..POOL_SIZE {
            for dst in 0..POOL_SIZE {
                if src != dst {
                    let hashed = hash(mixer[src], &mut hash_const);
                    mixer[dst] = mix(mixer[dst], hashed);
                }
            }
        }
        // Entropy beyond the pool size folds in afterwards. Unreachable for a
        // single-word seed, kept so the function is numpy's and not a subset.
        for &extra in entropy.iter().skip(POOL_SIZE) {
            for dst in 0..POOL_SIZE {
                let hashed = hash(extra, &mut hash_const);
                mixer[dst] = mix(mixer[dst], hashed);
            }
        }
        mixer
    }

    fn generate_state_u32(pool: &[u32; POOL_SIZE], words: usize) -> Vec<u32> {
        let mut hash_const = INIT_B;
        (0..words)
            .map(|i| {
                let mut v = pool[i % POOL_SIZE] ^ hash_const;
                hash_const = hash_const.wrapping_mul(MULT_B);
                v = v.wrapping_mul(hash_const);
                v ^ (v >> XSHIFT)
            })
            .collect()
    }

    /// `generate_state(n, dtype=uint64)`. numpy generates `2n` 32-bit words and
    /// views them as `uint64` — little-endian, so the LOW word comes first.
    pub fn generate_state_u64(pool: &[u32; POOL_SIZE], n: usize) -> Vec<u64> {
        let words = generate_state_u32(pool, 2 * n);
        (0..n)
            .map(|i| (u64::from(words[2 * i + 1]) << 32) | u64::from(words[2 * i]))
            .collect()
    }
}

/// The 128-bit LCG multiplier, `PCG_DEFAULT_MULTIPLIER_128`.
const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// numpy's `PCG64` bit generator, seeded through `SeedSequence`.
///
/// This is the `XSL-RR 128/64` variant: the 128-bit state is folded to 64 bits
/// by xor, then rotated by a count taken from the state's own top bits.
pub struct NumpyRng {
    state: u128,
    increment: u128,
    /// `has_uint32` / `uinteger`: the high half of the last 64-bit draw that
    /// `next_uint32` split, waiting to be returned by the next one.
    buffered_u32: Option<u32>,
}

impl NumpyRng {
    /// `np.random.default_rng(seed)` for a non-negative integer seed.
    ///
    /// The callers here all pass `_stable_seed(...)`, which is masked to 31
    /// bits, so the entropy is a single 32-bit word.
    pub fn new(seed: u32) -> Self {
        let pool = seed_sequence::mix_entropy(&[seed]);
        let state_words = seed_sequence::generate_state_u64(&pool, 4);
        let initial_state = (u128::from(state_words[0]) << 64) | u128::from(state_words[1]);
        let initial_sequence = (u128::from(state_words[2]) << 64) | u128::from(state_words[3]);

        // `pcg64_srandom_r`: zero the state, set the increment odd, step, add
        // the seed, step again. The two steps are not decoration -- dropping
        // either one shifts the whole stream.
        let mut rng = Self {
            state: 0,
            increment: (initial_sequence << 1) | 1,
            buffered_u32: None,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(initial_state);
        rng.step();
        rng
    }

    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_MULT)
            .wrapping_add(self.increment);
    }

    /// One 64-bit output. numpy steps **before** reading, not after.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let state = self.state;
        let folded = ((state >> 64) as u64) ^ (state as u64);
        let rotation = (state >> 122) as u32;
        folded.rotate_right(rotation)
    }

    /// `next_double`: the top 53 bits scaled by `2^-53`.
    ///
    /// The shift is 11 and the divisor is `9007199254740992.0`; taking the
    /// bottom 53 bits instead, or dividing by `2^64`, both produce a plausible
    /// uniform and a different stream.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `Generator.uniform(low, high)`, which numpy evaluates as
    /// `low + (high - low) * next_double()` -- not `lerp`, and not
    /// `low * (1 - t) + high * t`. The spelling is observable in the low bits.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next_f64()
    }

    /// `Generator.random()`: one `next_double`.
    pub fn random(&mut self) -> f64 {
        self.next_f64()
    }

    /// PCG64's `next_uint32`: the **low** half of a fresh 64-bit draw, with
    /// the high half kept for the next call.
    ///
    /// `next_uint64` and `next_double` neither read nor clear that buffer, so
    /// a stream that interleaves `integers` with `uniform` depends on it: a
    /// draw's second half can surface after any number of doubles.
    pub fn next_u32(&mut self) -> u32 {
        if let Some(high) = self.buffered_u32.take() {
            return high;
        }
        let next = self.next_u64();
        self.buffered_u32 = Some((next >> 32) as u32);
        next as u32
    }

    /// `buffered_bounded_lemire_uint32`: a uniform integer in `0..=range`,
    /// by Lemire's multiply-shift with rejection.
    fn bounded_lemire_u32(&mut self, range: u32) -> u32 {
        debug_assert!(range != u32::MAX);
        let range_excl = u64::from(range) + 1;
        let mut m = u64::from(self.next_u32()) * range_excl;
        let mut leftover = m as u32;
        if u64::from(leftover) < range_excl {
            let threshold = ((u64::from(u32::MAX) - u64::from(range)) % range_excl) as u32;
            while leftover < threshold {
                m = u64::from(self.next_u32()) * range_excl;
                leftover = m as u32;
            }
        }
        (m >> 32) as u32
    }

    /// `Generator.integers(low, high)` for a scalar `int64` in `[low, high)`.
    ///
    /// numpy's `_rand_int64` path: a range of zero returns `low` without
    /// drawing; a range that fits 32 bits draws through `next_uint32` (and
    /// its buffer), by Lemire's method unless it is exactly `2^32 - 1`.
    /// Wider ranges are not used by any caller and are refused rather than
    /// guessed at.
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        assert!(high > low, "integers needs low < high, got {low}..{high}");
        let range = (high - 1).wrapping_sub(low) as u64;
        if range == 0 {
            return low;
        }
        assert!(
            range <= u64::from(u32::MAX),
            "64-bit integer ranges are not reproduced"
        );
        let draw = if range == u64::from(u32::MAX) {
            self.next_u32()
        } else {
            self.bounded_lemire_u32(range as u32)
        };
        low.wrapping_add(i64::from(draw))
    }
}
