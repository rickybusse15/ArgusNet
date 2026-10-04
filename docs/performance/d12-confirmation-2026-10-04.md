# D-12 confirmation run, 2026-10-04

**Platform:** linux-x86_64 (glibc 2.39, 4 cores). **Build:** release
`argusnet-rs` at `1d65658` (main, with D-12 declared). **Reference:** live,
recorded into the stored reference ([`p49c-reference.json`](p49c-reference.json),
D-10) by these runs, under the same code hash. **Seeds:** `--seed-set
confirmation`, 30 seeds fixed in the commit that declared D-12 and run by
nobody before this. Raw report, with every run's numbers and per-metric
`statistics` (n, mean, median and sd per side, standard error of the
difference): [`d12-confirmation-linux-x86_64.json`](d12-confirmation-linux-x86_64.json).

D-12 ([PHASE_ACCEPTANCE.md](../migration/PHASE_ACCEPTANCE.md)) judges
`covariance_reduction_mean` by its cross-seed **median**, with the envelope
unchanged at 0.10 and D-1's five-qualifying-track minimum. This is the run that
judges that decision.

## Result: 7/7 scenarios pass

| Scenario | Verdict | `localisation_rmse_m` ref / Rust (mean) | `covariance_reduction_mean` ref / Rust (median, judged) | same, mean (not judged) |
| --- | --- | --- | --- | --- |
| baseline_coverage | pass | unavailable | unavailable | unavailable |
| intercept_stress | pass | 27.8 / 28.2 (Δ 0.44, SE 2.2) | 0.889 / 0.911 (Δ 0.022) | 0.870 / 0.856 (SE 0.038) |
| persistent_long | pass | unavailable | unavailable | unavailable |
| search_acquisition | pass | 41.7 / 40.1 (Δ −1.59, SE 3.1) | 0.844 / 0.815 (Δ −0.029) | 0.720 / 0.526 (SE 0.14) |
| planner_adversarial | pass | unavailable | unavailable | unavailable |
| sar_person_search | pass | 40.2 / 37.6 (Δ −2.68, SE 6.4) | 0.747 / 0.795 (Δ 0.048) | 0.427 / 0.424 (SE 0.20) |
| industrial_asset_survey | pass | unavailable | unavailable | unavailable |

Every exact check (frames, node and target ids, objective totals), every other
statistical metric, verdict agreement (30/30 seeds on every scenario) and the
timing gates pass. No run errored or was dropped.

## Reading it

1. **The medians agree within a third of the envelope on all three scored
   scenarios** (largest |Δ| 0.048 against 0.10). The A3.2 failures were the
   mean's tails, as D-12 supposed.
2. **The means would still have failed `search_acquisition`** (Δ −0.194 > 0.10,
   SE 0.14): a few far-negative seeds on the Rust side pull its mean down while
   its median sits with the reference's. Reported as found; the statistic was
   declared before this run and is not revisited on its result.
3. **Timing is a conservative reading.** The runs used 4 parallel jobs and
   shared the machine with unrelated debug builds; the worst tick p95 was
   4.44 ms (`intercept_stress`) against the 5 ms gate. A quiet single-job run
   measured well under that in A3.2. Worth watching: it is the gate with the
   least margin.

## Not covered

Unchanged from A3.2: profiles beyond the canonical scenarios, and
cross-platform determinism (A3.3, needs a macOS runner, D-9).
