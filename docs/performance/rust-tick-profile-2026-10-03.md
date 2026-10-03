# Rust tick profile, 2026-10-03 (P4.10-pre)

The first per-stage profile of the Rust tick, the measurement the P4.10
scheduler and parallel work are to be chosen from
([DELIVERY_PLAN.md](../migration/DELIVERY_PLAN.md) P4.10; the agent execution
and parallel tick plans proposed in #134 call it P4.10-pre). It replaces the
Python-era profile (localizer 40 %, VIO 13 %) as the basis for those
decisions; that profile described a different runtime.

**How:** `scripts/profile_rust_tick.py --runs 5 --seeds 7` on `main` at
`3a0ca89`, release `argusnet-rs sim --config --stage-timing`, every canonical
scenario, five runs each. Raw data:
[`rust-tick-profile-2026-10-03.json`](rust-tick-profile-2026-10-03.json).

**Machine:** cloud container, Intel Xeon @ 2.80 GHz, 4 vCPUs, Linux
6.18 x86_64, glibc 2.39, nothing else running. It is not a named reference
machine: the shares are indicative, and absolute times will differ
elsewhere. One seed per scenario; other seeds were not profiled.

## Whole tick

Legacy step 0.25 s. "Real time" is simulated time per wall time over the tick
loop alone (startup and replay writing excluded).

| Scenario | Drones | Ticks | Tick mean ± sd over runs (ms) | p95 | p99 | Real time | Startup (ms) | Replay write (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_coverage | 2 | 241 | 0.122 ± 0.010 | 0.160 | 0.268 | ~2050x | 122 | 1.0 |
| intercept_stress | 3 | 361 | 0.669 ± 0.080 | 1.011 | 1.113 | ~370x | 145 | 2.6 |
| persistent_long | 5 | 1201 | 0.559 ± 0.030 | 0.866 | 1.038 | ~450x | 262 | 4.9 |
| search_acquisition | 3 | 361 | 0.422 ± 0.053 | 0.528 | 0.657 | ~590x | 122 | 2.0 |
| planner_adversarial | 4 | 481 | 0.216 ± 0.031 | 0.294 | 0.375 | ~1160x | 109 | 1.7 |
| sar_person_search | 3 | 361 | 0.157 ± 0.040 | 0.188 | 0.256 | ~1590x | 153 | 1.9 |
| industrial_asset_survey | 2 | 481 | 0.127 ± 0.003 | 0.196 | 0.233 | ~1970x | 105 | 1.4 |

Run-to-run spread of the mean tick is 2–25 %. A speed-up smaller than that
cannot be claimed from one machine.

## Where the time goes

Share of summed stage time, the stages above 5 % in any scenario. The
Amdahl bound is the whole-tick speed-up if that stage cost nothing:
`1 / (1 - share)`.

| Scenario | 1st | 2nd | 3rd |
| --- | --- | --- | --- |
| baseline_coverage | mapping + localization 64.5 % (2.82x) | mission transitions 20.4 % | fusion 5.0 % |
| intercept_stress | **observations 79.2 % (4.81x)** | controllers + battery 10.5 % | fusion 7.7 % |
| persistent_long | mapping + localization 46.1 % (1.86x) | **mission transitions 45.6 % (1.84x)** | controllers + battery 2.8 % |
| search_acquisition | **observations 79.3 % (4.84x)** | controllers + battery 11.0 % | fusion 6.8 % |
| planner_adversarial | **mapping + localization 72.8 % (3.67x)** | mission transitions 12.6 % | controllers + battery 4.7 % |
| sar_person_search | observations 44.2 % (1.79x) | controllers + battery 29.9 % | safety monitor 9.5 % |
| industrial_asset_survey | mapping + localization 63.8 % (2.76x) | mission transitions 19.4 % | safety monitor 6.2 % |

Deconfliction, mission overrides, VIO/pose, command application and the
snapshot commit are each under 2 % everywhere.

## What it means for P4.10

1. **The tick is already far faster than real time** on these fleets (2–5
   drones): 370x at worst. Parallelism inside a tick buys latency, not
   feasibility, until fleets or sensor rates grow. That supports the plan's
   order: work reduction (B1) first, and parallelise only where this table
   predicts a gain larger than the 2–25 % run-to-run noise.
2. **Observation generation** is the one large, naturally parallel cost (79 %
   in the tracking scenarios; ADR-006 already keys its draws per
   `(node, target, tick)`). It is the first B3 (fork-join) candidate. Its
   absolute cost, 0.3–0.5 ms a tick, is small enough that per-task overhead
   matters: measure a coarse split (per node) before a fine one.
3. **Mission transitions cost 46 % of `persistent_long`** (0.22 ms a tick)
   and 13–20 % of the other scan scenarios. Stage 9 has little arithmetic of
   its own, so this looks like recomputing coverage or completion over the
   whole grid every tick. It is a **work-reduction (B1) target**, not a
   parallelism one: find what it recomputes, and cache or make it
   incremental. That is the next slice.
4. **Mapping + localization** (46–73 % in the scan scenarios) is the second
   B1 target: dirty tiles and summed-area coverage
   ([COVERAGE_AGGREGATES.md](../migration/COVERAGE_AGGREGATES.md)), then per-
   platform localizer fork-join if it is still large.
5. Controllers + battery (10–30 %) and the safety monitor (≤ 10 %) are
   per-platform and cheap. Not worth splitting at these fleet sizes.

Not measured here: allocation counts and the work counters (observation
pairs, footprints, tiles touched). They are the next B0 sub-slice, and B1
needs them to show that work actually dropped.
