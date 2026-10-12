# B2.4: the mission lane at 1 Hz, 2026-10-12

**Platform:** linux-x86_64 (glibc 2.39, 4 cores). **Build:** release
`argusnet-rs` on `claude/scheduler-lane-rates`. **Run under test:**
`argusnet-rs sim --executor lanes --mission-rate 1/1` (`scheduled_v1`): the
mission lane at 1 Hz instead of every tick, through
`scripts/run_phase_acceptance.py --rust-arg ...`. **Reference:** the stored
reference (D-10), envelopes unchanged.

## Judged: D-12 confirmation seeds (30), 7/7 pass

| Scenario | Verdict | `localisation_rmse_m` ref / Rust (mean) | `covariance_reduction_mean` ref / Rust (median, judged) |
| --- | --- | --- | --- |
| baseline_coverage | pass | unavailable | unavailable |
| intercept_stress | pass | 27.8 / 28.2 | 0.889 / 0.910 |
| persistent_long | pass | unavailable | unavailable |
| search_acquisition | pass | 41.7 / 40.1 | 0.844 / 0.815 |
| planner_adversarial | pass | unavailable | unavailable |
| sar_person_search | pass | 40.2 / 37.6 | 0.747 / 0.795 |
| industrial_asset_survey | pass | unavailable | unavailable |

These are the D-12 confirmation run's numbers
([d12-confirmation-2026-10-04.md](d12-confirmation-2026-10-04.md)) to the
digit shown. Raw report:
[`b24-mission-1hz-confirmation-linux-x86_64.json`](b24-mission-1hz-confirmation-linux-x86_64.json).

## Canonical seeds (5): the same verdicts as the legacy rate

On the five canonical seeds both the legacy rate and 1 Hz give 6/7, with
`sar_person_search` failing identically (`localisation_rmse_m` |Δ| 11.29 against
11.18; `covariance_reduction_mean` median 0.488 against 0.703). That failure is
the legacy run's, not this change's: A3.2 found five seeds cannot resolve these
metrics on this scenario, and D-12 makes the 30 confirmation seeds the judge
([a32-qualification-2026-10-03.md](a32-qualification-2026-10-03.md)).

Run for run, every scored metric is the same at 1 Hz as at the legacy rate
except `energy_reserve_min` on `persistent_long` and `planner_adversarial`,
which moves below the fourth decimal. Frames and node/target identities are
equal on every run.

## Reading

Running the mission stage four times less often (0.25 s ticks, 1 Hz) does not
move the scored outcomes: its inputs are cumulative, phase transitions land at
most 0.75 s later, and inspect dwell is credited with the elapsed time. What
the scheduler buys here is the mechanism -- a lane off the legacy rate with
the skips traced, elapsed time carried and the lag checkpointed -- not a
behaviour difference. The legacy rate stays the default; P4.10f decides rates.
