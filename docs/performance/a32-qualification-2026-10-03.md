# A3.2 statistical qualification, 2026-10-03

**Platform:** linux-x86_64 (glibc 2.39, 4 cores). **Build:** release
`argusnet-rs`. **Reference:** live, recorded into the stored reference
([`p49c-reference.json`](p49c-reference.json), D-10) by these runs.

Per [VERIFICATION_AND_ROLLOUT.md](../migration/VERIFICATION_AND_ROLLOUT.md) §3
and the agent plan's A3.2: 30 predetermined seeds per canonical scenario for
qualification, and a separate set of 10 held-out seeds for the final
acceptance, both fixed in `scripts/run_phase_acceptance.py`
(`QUALIFICATION_SEEDS`, `HELD_OUT_SEEDS`: drawn once from named generators,
disjoint from each other and from the five canonical seeds). The envelopes are
the ones declared in [PHASE_ACCEPTANCE.md](../migration/PHASE_ACCEPTANCE.md),
unchanged. Raw reports, with every run's numbers and per-metric variance
estimates (`statistics`: n, mean and sd per side, standard error of the
difference): [`a32-qualification-linux-x86_64.json`](a32-qualification-linux-x86_64.json),
[`a32-heldout-linux-x86_64.json`](a32-heldout-linux-x86_64.json).

## Results

| Set | Scenario | Verdict | `localisation_rmse_m` ref / Rust | `covariance_reduction_mean` ref / Rust |
| --- | --- | --- | --- | --- |
| qualification (30 seeds) | baseline_coverage | pass | unavailable | unavailable |
| qualification (30 seeds) | intercept_stress | pass | 27.6 / 25.6 (Δ -1.99, SE 3) | 0.861 / 0.9 (Δ 0.0389, SE 0.038) |
| qualification (30 seeds) | persistent_long | pass | unavailable | unavailable |
| qualification (30 seeds) | search_acquisition | fail: covariance_reduction_mean | 35.5 / 35.9 (Δ 0.416, SE 3.2) | 0.834 / 0.73 (Δ -0.104, SE 0.074) |
| qualification (30 seeds) | planner_adversarial | pass | unavailable | unavailable |
| qualification (30 seeds) | sar_person_search | fail: covariance_reduction_mean | 35.8 / 37 (Δ 1.21, SE 4.7) | 0.55 / 0.389 (Δ -0.161, SE 0.2) |
| qualification (30 seeds) | industrial_asset_survey | pass | unavailable | unavailable |
| held-out (10 seeds) | baseline_coverage | pass | unavailable | unavailable |
| held-out (10 seeds) | intercept_stress | fail: covariance_reduction_mean | 28.4 / 25.6 (Δ -2.72, SE 3.3) | 0.783 / 0.901 (Δ 0.118, SE 0.083) |
| held-out (10 seeds) | persistent_long | pass | unavailable | unavailable |
| held-out (10 seeds) | search_acquisition | fail: covariance_reduction_mean | 32.3 / 38.9 (Δ 6.58, SE 4.3) | 0.719 / 0.566 (Δ -0.153, SE 0.19) |
| held-out (10 seeds) | planner_adversarial | pass | unavailable | unavailable |
| held-out (10 seeds) | sar_person_search | fail: covariance_reduction_mean | 34.6 / 33.6 (Δ -0.946, SE 6.6) | 0.577 / 0.445 (Δ -0.131, SE 0.21) |
| held-out (10 seeds) | industrial_asset_survey | pass | unavailable | unavailable |
Every exact check (frames, node and target ids, objective totals), every other
statistical metric (track continuity, reacquisition time, mission completion,
energy reserve) and verdict agreement pass on all 7 scenarios in both sets. No
run errored or was dropped. The timing gates pass although the runs were made
with 4 parallel jobs (worst tick p95 well under 5 ms), so they are a
conservative reading.

## Findings

1. **`sar_person_search`'s localisation RMSE was five-seed noise.** The
   canonical five failed by 0.11 m; over 30 seeds the difference is 1.21 m
   (standard error 4.7 m), and 0.95 m on the held-out ten. It passes on both.
2. **`covariance_reduction_mean` fails three scenarios, but not in one
   direction, and never by more than 1.5 standard errors.** Rust is higher
   than the reference on held-out `intercept_stress` and lower on
   `search_acquisition` and `sar_person_search`; on qualification
   `intercept_stress` it passes. The metric is heavy-tailed: most seeds land at
   0.7–0.95, a few far negative on either side (reference to −1.36, Rust to
   −3.29 on `sar_person_search`), and `sar_person_search` scores one track per
   seed. Its standard error (0.07–0.21) is of the order of, or above, the
   declared 0.10 envelope, so a mean over these seed counts cannot pass it even
   at parity. This is a measurement-power finding, not a parity defect.

## Not widened; decision D-12

The envelope is **not** changed after seeing these results. Whether
`covariance_reduction_mean` should be judged by a robust statistic (median or
a trimmed mean, declared before the next run), by a larger pre-declared seed
count, or by an envelope sized from its measured variance, is decision **D-12**
in the agent plan's register, for the maintainer.

## Not covered

The profile list in VERIFICATION_AND_ROLLOUT.md §3 beyond the canonical
scenarios (occlusion-aware mapping, frontier, VIO substitution, low
battery/egress, dense obstacles, weather) is not run: those profiles are not
canonical scenarios yet. Cross-platform determinism is A3.3 (needs a macOS
runner, D-9).
