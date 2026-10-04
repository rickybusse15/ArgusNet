# B3: the localizer's particle work forked per platform

Second B3 candidate (PARALLEL_TICK_PLAN.md §4.2). The Rust tick profile puts
mapping + localization at 64–73 % of tick time in `baseline_coverage`,
`planner_adversarial` and `industrial_asset_survey`.
Machine: Linux x86_64, 4 vCPU, release build, 2026-10-04.

## Where stage 8's time goes

The stage was instrumented temporarily (not merged). Per-tick means, seed 7:

| Scenario | particle draws | footprint scoring | localizer update | rasterize | grid copy | stage total |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_coverage | 28.2 µs | 14.7 µs | 4.8 µs | 1.4 µs | 1.3 µs | 52 µs |
| persistent_long | 65.7 µs | 91.2 µs | 11.5 µs | 8.3 µs | 9.4 µs | 191 µs |

Particle draws and scoring are 80–85 % of the stage. Each drone draws 200
particles, two normals each. The draws cannot be made cheaper without a
behaviour change, because their values are pinned: Box–Muller through `libm`,
guarded by fixtures. They can be forked, though: each drone's draws come from
its own `(platform, step)` stream, and its scoring reads only the coverage
counts.

## RNG audit (B3 step 3)

No shared stream needs fixing before forking:

| Stream | Keying | Forked? |
| --- | --- | --- |
| Observation pairs | `(node, target, tick)` | yes (#190) |
| Clutter | `(node, "clutter", tick)` | no: per sensor, after the pairs |
| Sensor bias walk | one stream per sensor, continuous | no: stateful, stepped serially in pair order |
| Localizer particles | `("particles", platform, step)` | yes (this change) |
| VIO / pose | no random draws | — |
| `argusnet-localization`, `argusnet-mapping` | no random draws | — |

## What changed

- Stage 8 computes each platform's inputs serially. Each platform's offsets and
  scores are then computed on the session's pool. `GridLocalizer::update`
  stays serial in platform order, so a failure stops at the same platform as
  before.
- The fork is gated on fleet size: `MIN_PLATFORMS_TO_FORK = 12`. Below that,
  the serial path runs whatever `--workers` says.
  `ComputePool::map_ordered_from` takes this per-call minimum.

## Determinism

- `replay_hashes.py --check` against `main`: 14/14 identical at 1, 2 and 4
  workers.
- `--workers 4 --lifecycle`: all equal.
- New `tests/compute_workers.rs::a_forked_localizer_matches_the_serial_one`:
  a 16-drone scan mission (fixture `scenario_config_scan_16_drones.json`)
  takes the forked path. It asserts that at least 12 platforms are localized
  at once, and that every tick's estimates at 2 and 4 workers are identical
  to the serial run's.

## Speed

**Ungated** (every fleet forked): the canonical scenarios got much slower.

| Scenario | tick mean, 1 worker | 2 workers | 4 workers |
| --- | --- | --- | --- |
| baseline_coverage | 0.072 ms | 0.158 ms | 0.203 ms |
| planner_adversarial | 0.136 ms | 0.222 ms | 0.309 ms |
| industrial_asset_survey | 0.089 ms | 0.181 ms | 0.220 ms |
| persistent_long | 0.246 ms | 0.316 ms | 0.363 ms |

The whole stage costs about 50 µs at this size. Waking pool workers that sleep
between ticks costs about as much, so forking only adds latency. The
`b3-mapping-ungated-*` JSON files hold these runs.

**Gated** at 12 platforms. Scan missions, 120 s, seed 7, 5 runs per cell:

| Drones | Workers | Tick mean (median) | Run range | p95 | Speed-up |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 0.211 ms | 0.203–0.239 | 0.307 | 1.00x |
| 4 | 4 | 0.221 ms | 0.205–0.293 | 0.330 | 0.95x (serial path; within noise) |
| 8 | 1 | 0.409 ms | 0.391–0.438 | 0.604 | 1.00x |
| 8 | 4 | 0.440 ms | 0.396–0.477 | 0.619 | 0.93x (serial path; within noise) |
| 16 | 1 | 0.767 ms | 0.747–0.781 | 1.060 | 1.00x |
| 16 | 2 | 0.675 ms | 0.641–0.744 | 0.933 | 1.13x |
| 16 | 4 | 0.645 ms | 0.587–0.676 | 0.912 | 1.19x |
| 32 | 1 | 1.643 ms | 1.564–1.718 | 2.294 | 1.00x |
| 32 | 2 | 1.502 ms | 1.438–1.514 | 1.929 | 1.09x |
| 32 | 4 | 1.093 ms | 1.079–1.126 | 1.444 | 1.50x |

From 16 drones the gain beats run-to-run noise: the run ranges at 1 and 4
workers do not overlap. It is smaller than observation generation's (2.7x at
24 drones), because each platform's task is only 20–40 µs.

## Candidates not forked, and why

- **Footprint rasterization** (B3 candidate 3): 1.4–8 µs per tick, below the
  cost of waking a worker.
- **Controllers and battery** (candidate 4): not attempted yet. They are
  10–30 % of tick time in the tracking and SAR scenarios, but per-platform
  work at canonical sizes is a few microseconds. On this evidence they would
  need a fleet far beyond today's to pay. The next step is to measure them
  with the work counters P4.10-pre still lacks.
- **Pool wake-up cost** is the common limit. A pool that spins briefly
  between ticks, instead of sleeping, would lower every crossover point, at
  the cost of burning idle CPU. That is a separate, measured decision.
