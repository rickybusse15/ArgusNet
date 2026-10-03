# P4.9c: phase acceptance for the Rust CLI

**Status:** envelopes declared 2026-09-23, before any acceptance run. Reaching
a pass for the five scenarios the builder then supported took two defect
fixes and no widening; see [Results](#results). Since P4.9d all seven are
scored. Six pass every envelope, and `sar_person_search` fails one,
`covariance_reduction_mean`; see
[After P4.9d](#after-p49d-all-seven-scenarios-scored) and
[STATUS.md](STATUS.md).

DELIVERY_PLAN P4.9 closes Phase 4 with: *"Run the canonical scenario suite,
edge profiles and multi-seed statistical tests. Emit resolved
config/version/provenance and record startup/tick distributions. ... Exit:
Rust CLI reproduces supported scenarios within approved envelopes, no Python
runtime dependency, direct fusion parity and meaningful perf gates."*

## What runs

The suite is `argusnet.evaluation.scenarios.CANONICAL_SCENARIOS`, the same
seven scenarios `argusnet benchmark` runs, on its five `CANONICAL_SEEDS`
(7, 42, 137, 9999, 31415). For each scenario and seed:

1. **Reference.** `run_config_with_sim`: `argusnet sim` with the scenario's
   flags. The replay is scored by `evaluate_replay` and `check_pass_fail`.
2. **Rust.** The same flags through `argusnet sim --export-config`, the P4.9
   adapter that translates legacy flags into a versioned config. Then
   `argusnet-rs sim --config` on a **release** build, scored the same way.
   Nothing Python runs on this side after the config is written.

Output-only flags (`--streaming`, `--max-frames-in-memory`) do not change the
scenario, so they are dropped from the Rust side's config. A scenario the
native builder refuses (a mission profile) is recorded as unsupported, with
the builder's reason, and is not scored.

`scripts/run_phase_acceptance.py` runs the suite and writes
`docs/performance/p49c-acceptance.json`.

## Declared envelopes

Declared here before the first run, and not to be widened after one. A
failure is investigated as a defect first.

**Exact, per seed.** Anything no random draw reaches must match:

| Check | Why it is exact |
| --- | --- |
| frame count | run length is configuration |
| node ids and target ids | the scenario builder is proven structurally identical (P4.9b) |
| `required_objectives_total` | scenario configuration |

**Statistical, per scenario, over the five seeds.** Stage 6 draws from keyed
ChaCha streams, not numpy's (ADR-006). So which observations are accepted,
what fusion makes of them, and what the localizer concludes all legitimately
differ. Each metric is compared as the difference of seed means, where both
sides report it:

| Metric | Envelope on \|mean(Rust) − mean(reference)\| |
| --- | --- |
| `localisation_rmse_m` | ≤ max(25% of the reference mean, 2.0 m) |
| `track_continuity_mean` | ≤ 0.10 |
| `time_to_reacquire_mean_s` | ≤ max(50% of the reference mean, 2.0 s) |
| `covariance_reduction_mean` | ≤ 0.10 |
| `mission_completion_rate` | ≤ 0.20 |
| `energy_reserve_min` | ≤ 0.02 |

**Amended by D-1 (2026-10-03).** `covariance_reduction_mean` measures each
track at its last update, not its last appearance, and is scored only when
each side has at least five qualifying tracks across its seeds. Neither side
reaching five makes the check *unavailable*; only one side reaching it fails.
A statistical metric neither side reports is *unavailable*, never a pass, and
unavailable checks are listed beside each verdict. The envelope values are
unchanged.

**Verdict.** `check_pass_fail` gives the same pass or fail on at least four of
the five seeds.

**Performance.** Release build, on the machine the run records:

| Gate | Limit |
| --- | --- |
| tick time p95 | ≤ 5 ms in every canonical scenario |
| startup (config to first tick, world build included) | ≤ 2 s |
| speed-up over the reference's simulation wall clock | ≥ 5x |

The speed-up compares like with like. The reference's `wall_clock_s` times
its simulation loop, and scenario building happens before it. The Rust side
of the comparison is therefore its tick loop, total time less startup.
Startup, which includes the world build, is gated separately.

## Direct fusion parity

The Rust tick fuses in process, through `argusnet-core`'s `TrackingEngine`.
The reference fuses through `argusnetd` over gRPC. Until this slice, nothing
compared the two paths: P4.8c's fusion-stage tests exercise the stage, not the
daemon.

`tests/test_direct_fusion_parity.py` records, from a real reference run,
every `IngestFrameRequest` the simulation sends and every
`IngestFrameResponse` the daemon returns, along with the tracker
configuration the daemon was given. It then replays the requests through
`argusnet-rs fuse`, a hidden subcommand. `fuse` performs `argusnetd`'s proto
conversion and `TrackingEngine::ingest_frame` in process and writes the
responses. **Every response must be the same protobuf message**, compared
after decoding, because protobuf map fields have no fixed byte order. Over 60
ticks of a four-station, three-drone, two-target run, every frame matches.

That settles the fusion arithmetic. The acceptance run covers what comes
after it: observation streams that differ by design (ADR-006) must still
score within the envelopes above.

## Results

Run on 2026-09-23 on macOS 27.0 arm64 (Python 3.12.13), with a release
`argusnet-rs`. Five seeds per scenario. The report is
`docs/performance/p49c-acceptance.json`.

### How the pass was reached

The envelopes above are the ones declared before the first run, unchanged.
The suite ran three times:

1. **First run: search_acquisition failed** `localisation_rmse_m`. Reference
   mean 77.6 m, Rust 41.5 m, |Δ| 36.2 m against 19.4 m allowed. Rust was lower
   on four seeds of five, so this was not seed noise. The reference's error
   tail (p99 427 m against Rust's 146 m) came from tracks sitting at exactly
   (0, 0, 0) with zero velocity. Meanwhile every observation ray for the
   target passed within metres of it.

   **Defect 1, in `argusnet-core` (both runtimes):** `IMMTrack3D::update_position`
   normalised the mode probabilities by `max(sum, 1e-12)`. When both models'
   likelihoods underflowed on a large innovation, the probabilities fell to
   about 1e-20 instead of summing to one. `state()` blends the filters by them,
   so the track read as the origin. The next `predict` mixed both sub-filters
   by them too, which zeroed the filters for good, and the track stayed at the
   origin until it was deleted. Both runtimes share the engine and both
   collapsed tracks (10 track-frames in the reference, 3 in Rust, on
   search_acquisition seed 42). The reference simply collapsed more often.
   Fixed by dividing by the true sum. The likelihood floors keep that sum
   above 5e-31, and `f64::max` discards NaN. When both floors bind they
   cancel, so the prior stands.
   `imm_survives_an_innovation_both_models_reject` fails on the old code.

2. **Second run: search_acquisition passed; intercept_stress failed**
   `localisation_rmse_m`. Reference 47.7 m, Rust 62.1 m, |Δ| 14.4 m against
   11.9 m allowed. In every seed's RMSE, only one track was scored.

   **Defect 2, in `evaluation/metrics.py`:** `compute_localisation_rmse`
   paired tracks with truths only in the first frame that held *any* track,
   by nearest distance. A track confirmed later was never scored. A track
   that happened to pass another target at that instant was scored against
   that target for the whole run: the reference's seed-42 RMSE of 117 m was
   `asset-lynx`'s track measured against `asset-fox`. Now each track is paired
   in its own first frame. A track named for a truth target (the fusion
   service keys tracks by the observations' target ID) is paired with that
   target. Two new tests fail on the old code.

3. **Third run: every scored scenario passes.**

Both fixes changed the reference as well as Rust: the reference fuses through
the same `argusnet-core`, and both sides are scored by the same metric. The
fixes corrected the measurement; no envelope moved.

### Per scenario (third run)

| Scenario | Exact | `localisation_rmse_m` ref / Rust (allowed \|Δ\|) | Other statistical | Verdict | Worst tick p95 | Worst startup | Speed-up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_coverage | 20/20 | — (no targets) | all within | 5/5 | 0.045 ms | 32 ms | 24.8x |
| intercept_stress | 20/20 | 25.3 / 31.0 m (≤ 6.3) | all within | 5/5 | 0.115 ms | 24 ms | 67.5x |
| persistent_long | 20/20 | — (no targets) | all within | 5/5 | 0.213 ms | 43 ms | 16.4x |
| search_acquisition | 20/20 | 34.1 / 40.3 m (≤ 8.5) | all within | 5/5 | 0.079 ms | 20 ms | 81.0x |
| planner_adversarial | 20/20 | — (no targets) | all within | 5/5 | 0.106 ms | 18 ms | 25.5x |
| sar_person_search | refused | | | | | | |
| industrial_asset_survey | refused | | | | | | |

The largest statistical deltas elsewhere: `covariance_reduction_mean` 0.044
(search_acquisition) and 0.030 (intercept_stress) against 0.10;
`track_continuity_mean` 0.022 against 0.10. Every other metric agrees to
0.02 or better. The worst single tick in any run was 0.34 ms
(persistent_long, 1,201 ticks).

### What the results do not show

* **Two canonical scenarios are unsupported.** `sar_person_search` and
  `industrial_asset_survey` are mission profiles. Both turn on
  `safety_blocking`, which the native builder refuses. The profiles
  otherwise set only ordinary options and descriptive metadata. The real gap
  is stage 4: the hard safety monitor was ported and tested in P4.8c, but it
  is not yet assembled from a bundle. That is the next slice, P4.9d.
* **The verdict check is weak evidence here.** `check_pass_fail` fails every
  seed on both sides, because `track_continuity_mean` sits under its
  threshold everywhere (0 in the scan scenarios, which have no targets). So
  5/5 agreement confirms only that the two sides fail the same way. This is
  a benchmark threshold matter; the port does not cause it.
* **intercept_stress passes with little margin**: 5.7 m against 6.3 m. One
  Rust seed (31415, 49.9 m) carries a track that ran off at 1.25 s. Its only
  two rays were 3.6° apart, above the tracker's 2.5° minimum. The resulting
  550 m triangulation was accepted with a 6 m measurement std. The reference
  admits the same geometry, and which seed meets it is a matter of
  observation draws (ADR-006). It is recorded in
  [KNOWN_GAPS.md](../KNOWN_GAPS.md) as tracker tuning, not changed here.
* **The Rust replay was missing each track's `lifecycle_state` and
  `quality_score`.** Found while tracing tracks, and fixed additively. Both
  fields are now declared in `docs/replay-schema.json`, and the committed
  Rust replay fixture was re-blessed deliberately.

### Verification

| Check | Result |
| --- | --- |
| `scripts/run_phase_acceptance.py` (release) | 5/5 scored scenarios pass; exit 0 |
| `tests/test_direct_fusion_parity.py` | every response the same message, over a 60-tick reference run and over hand-built frames carrying metadata, safety events and a non-default tracker config |
| `tests/test_phase_acceptance.py` | the judge on synthetic runs, at each declared boundary |
| Mutation round, Python (judge, metric, `--export-config`) | 20 mutants, 20 caught |
| Mutation round, Rust (IMM normalisation, replay fields, `fuse`, provenance, timing) | 16 mutants. 4 survived at first: `fuse` built with the default tracker config; `fuse` dropping safety events; p95 read as p50; odd-length hex. Tests were added for each (the hand-built parity case, `tick_summary` and `hex_decode` unit tests), and all 16 are now caught |
| `cargo test --workspace`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check` | 1085 passed, 0 failed; clean; clean |
| `ruff check`, `ruff format --check` (`src/argusnet scripts tests`) | clean |
| `pytest tests/` (`ARGUSNET_REQUIRE_RS=1`) | 1046 passed, 9 skipped, 0 failed |
| CI | **not run**: nothing is pushed |

After the third acceptance run, only tests changed, plus one refactor
(`tick_summary`) whose output is byte-identical. So the report stands without
another run.

## After P4.9d: all seven scenarios scored

[SAFETY_AND_PROFILES.md](SAFETY_AND_PROFILES.md) assembles stage 4 and builds
the mission profiles natively, so the two refused scenarios are now scored.
The report was rerun whole on the final build. **Six of seven scenarios pass
every envelope.** `industrial_asset_survey` passes. `sar_person_search` passes
everything except `covariance_reduction_mean` (reference −1.875, Rust −0.181,
against 0.10 allowed). That metric is decided by at most one track per seed,
reported on one reference seed against three Rust seeds, and it measures the
coast before a track's deletion. The analysis is in that doc. The envelope is
not widened; the metric's definition is an open decision.

## After D-1: the covariance metric redefined (2026-10-03)

Rerun of all seven scenarios with the D-1 metric and judge, on
linux-x86_64 (glibc 2.39, 4 cores, shared with builds). The report is
[`p49c-acceptance-linux-x86_64.json`](../performance/p49c-acceptance-linux-x86_64.json).
The macOS report above, `p49c-acceptance.json`, is kept as the record of the
old definition; this container cannot rerun it on that platform.

| Scenario | Verdict | Notes |
| --- | --- | --- |
| baseline_coverage, persistent_long, planner_adversarial, industrial_asset_survey | pass | RMSE, reacquisition and covariance **unavailable**: no targets, so neither side reports them (previously counted as passes) |
| intercept_stress | pass | covariance 0.893 / 0.905 over 20 / 20 tracks |
| search_acquisition | pass | covariance 0.834 / 0.864 over 15 / 15 tracks |
| sar_person_search | **fail**: `localisation_rmse_m` | covariance now **passes**: 0.544 / 0.495 over 5 / 5 tracks, \|Δ\| 0.049. RMSE: reference 44.72 m, Rust 33.42 m, \|Δ\| 11.29 against 11.18 allowed |

Every exact check and every performance gate passes (worst tick p95
0.344 ms, startup 149 ms, speed-up 116x, under build load).

**The RMSE failure is platform-dependent, and not widened.** On macOS the
same envelope passed (reference 39.4 m, Rust 33.0 m). The Rust side barely
moves between platforms; the Python reference moves 5 m, because its draws
are not platform-independent (numpy's `uniform` is fused on some builds; see
`scripts/capture_behaviors_fixture.py`) and stage 6 draws differ by design
(ADR-006). So an envelope judged against a reference recomputed on each
platform can flip with the platform. Whether acceptance should fix one
reference platform, or score Rust against a stored reference, is a decision,
recorded in [KNOWN_GAPS.md](../KNOWN_GAPS.md).

## The stored reference (D-10, 2026-10-03)

Acceptance now scores Rust against a **stored** reference by default:
[`p49c-reference.json`](../performance/p49c-reference.json), recorded once on
linux-x86_64 (seeded from the D-1 rerun above, whose reference code is
unchanged since but for a log line moved to stderr), tagged with its platform,
source commit and a SHA-256 of `src/argusnet`'s sources. A run whose
`src/argusnet` differs says the reference may be stale.
`--reference live` recomputes it on the machine at hand, and
`--reference live --record-reference` re-records it. The speed-up gate then
compares Rust's wall clock here with the reference's wall clock on the
recording machine; the report names that machine.

The stored-mode rerun ([`p49c-acceptance-stored.json`](../performance/p49c-acceptance-stored.json))
matches the live one: 6/7 pass, and every Rust number is unchanged, which also
confirms A1/A2 changed no behaviour. `sar_person_search` still fails
`localisation_rmse_m` (reference 44.72 m, Rust 33.42 m, |Δ| 11.29 against
11.18 allowed).

**That failure is inside five-seed noise, and the envelope is not widened.**
Per-seed RMSE, reference: 26.8, 47.4, 80.3, 28.5, 40.6 m (sd 21.7); Rust: 16.9,
42.1, 39.4, 27.1, 41.6 m (sd 11.1). The standard error of the difference of
means is 10.9 m, so |Δ| is about one standard error, carried by one reference
seed. Five seeds cannot tell this apart from parity; A3.2's ≥30 predetermined
seeds with a held-out set can, and decide it.

