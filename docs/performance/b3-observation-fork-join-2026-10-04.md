# B3: observation generation forked across workers

First fork-join candidate of PARALLEL_TICK_PLAN.md §4.2: observation generation,
which the Rust tick profile ([rust-tick-profile-2026-10-03.md](rust-tick-profile-2026-10-03.md))
put at 79 % of tick time in `intercept_stress` and `search_acquisition`.
Machine: Linux x86_64, 4 vCPU, release build, 2026-10-04.

## What changed

- `argusnet_sim::compute::ComputePool`: a session-owned, bounded worker pool
  (never rayon's global pool). Its only primitive, `map_ordered`, returns
  results in input order. Below 8 items it runs on the calling thread.
- Stage 6 is split in three. Per-sensor pose and look are computed serially.
  Every `(sensor, target)` pair is then evaluated as a pure function: the
  gates, line of sight, detection, and noise draws from the pair's own
  `(node, target, tick)` stream. Results are combined serially in pair order:
  tally, records, rejections, the per-sensor bias walk (stateful, so never
  forked), observations, then clutter.
- `argusnet-rs sim --workers N` (default **1**). The oracle and terrain traits
  now require `Sync`; every implementation already met it.

## Determinism

The replay is identical at every worker count:

- `scripts/replay_hashes.py --workers {1,2,4} --check` against `main`'s
  hashes: 14/14 identical at each count.
- `--workers 4 --lifecycle`: reset and resume are equal on every scenario
  and seed.
- `tests/compute_workers.rs`: the test bundle's replay at 2, 4 and 8 workers
  equals the serial one.
- The scaled fleets below: replays are byte-identical at 1 and 4 workers
  (8/16/8 and 24/48/24).

## Speed: the decision rule says no at today's fleet sizes

**Canonical scenarios**, 5 runs at 1 and at 4 workers
(`b3-observe-workers{1,4}-linux-x86_64.json`):

| Scenario | tick mean, 1 worker | tick mean, 4 workers | p95, 1 → 4 workers |
| --- | --- | --- | --- |
| intercept_stress | 0.433 ± 0.046 ms | 0.466 ± 0.038 ms | 0.655 → 0.696 ms |
| search_acquisition | 0.277 ± 0.015 ms | 0.356 ± 0.034 ms | 0.371 → 0.586 ms |
| sar_person_search | 0.097 ± 0.010 ms | 0.109 ± 0.014 ms | 0.141 → 0.186 ms |

These fleets give a few dozen pairs per tick, and the whole stage takes about
0.2–0.3 ms. Handing that work to threads costs more than it saves, so 4
workers are slower. Per §9 the default stays serial.

**Scaled fleets.** Target-tracking runs with drones / targets / ground
stations as listed, 60 s, seed 7, median of 3 runs:

| Fleet | 1 worker | 2 workers | 4 workers | Speed-up at 4 workers |
| --- | --- | --- | --- | --- |
| 4 / 8 / 4 | 0.462 ms | 0.499 ms | 0.438 ms | 1.06x (within noise) |
| 8 / 16 / 8 | 1.818 ms | 1.606 ms | 1.161 ms | 1.57x |
| 16 / 32 / 16 | 7.613 ms | 5.557 ms | 3.565 ms | 2.14x |
| 24 / 48 / 24 | 21.19 ms | 14.88 ms | 7.78 ms | 2.72x |

Pair work grows as sensors × targets. From about 8 drones and 16 targets the
gain exceeds the run-to-run noise, and at 24/48/24 tick p95 falls from 34.1 ms
to 9.7 ms.

## Decision

- The mechanism is merged; the default stays at 1 worker.
- `--workers N` is for large fleets.
- Choosing the worker count from fleet size automatically is deferred. It
  needs the work counters that are still open in P4.10-pre, so that the
  choice rests on measured pair counts rather than a guessed threshold.
