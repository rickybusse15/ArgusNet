# Python Runtime Baseline Profile

Measured 2026-09-02, before any Rust migration work, so that later phases have
something to compare against. This is the profile the migration plan's ordering
should follow — and it **contradicts two assumptions that were made before
measuring**. Those are called out below.

## Method

```bash
python -m cProfile -o sim_profile.prof -m argusnet sim \
  --map-preset regional --terrain-preset alpine --mission-mode scan_map_inspect \
  --drone-count 4 --ground-stations 7 --duration-s 30 --seed 7 --replay out.json
```

30 simulated seconds at `dt_s = 0.25` — 121 frames. Machine: Linux x86_64,
CPython 3.11.15. Absolute times are machine-specific; the *proportions* are the
durable result.

## Where the time goes

Total: **6.34 s** for a 121-frame run.

| Stage | Time | Share | Per frame |
|---|---:|---:|---:|
| Scenario construction (`build_default_scenario`) | 2.64 s | 42% | — (one-time) |
| Tick loop (`run_simulation`) | 3.27 s | 52% | 27 ms |
| Replay serialization (`write_replay_document`) | 0.33 s | 5% | — (one-time) |

### Scenario construction — 42%, and it is not per-tick

| Function | Time | Calls |
|---|---:|---:|
| `procedural.build_terrain_layer` | 1.69 s | 1 |
| ↳ `environment.TerrainLayer._build_pyramid` | 1.32 s | 8 |
| `procedural.build_land_cover_layer` | 0.91 s | 1 |
| ↳ `obstacles._point_in_polygon` | 0.77 s | 39,744 |

This is a startup cost, but not a harmless one: the viewer pays all of it every
time the operator presses **Run**, before a single frame renders. It is the
dominant cost of a short run.

> **Correction to the plan.** `_point_in_polygon` and `_point_on_segment`
> together account for ~1.1 s and look like an obstacle/collision hot path. They
> are not. `print_callers` attributes **39,744 of the 39,748** `_point_in_polygon`
> calls to `build_land_cover_layer` — scenario construction — and only **4** to
> the runtime `point_inside` collision check. Porting obstacle primitives for
> per-tick speed would optimise something that runs once.

### Tick loop — 27 ms per frame

| Component | Time | Share of tick loop | Per frame |
|---|---:|---:|---:|
| `localization.GridLocalizer.update` | 1.30 s | **40%** | 10.7 ms |
| `localization.vio.EKFVIO.process_image` | 0.43 s | 13% | 3.5 ms |
| gRPC round-trip (`IngestStream.ingest_frame`) | 0.21 s | 6% | 1.73 ms |
| `environment.TerrainLayer.height_at` | 0.30 s | 9% | 2.5 ms |

`GridLocalizer.update` is the single largest per-tick cost by a wide margin. It
scores 200 particles through a Python list comprehension calling
`_score_position` — 20,800 calls over the run, ~172 per frame, each slicing the
coverage grid.

> **Correction to the plan.** The plan assumed `EnvironmentQuery.los` would be
> "almost certainly the hottest leaf". It does not appear in the top 32 at all
> for this scenario, because occlusion-aware mapping is off by default. LOS
> raycasting is only hot under `--occlusion-aware-mapping`. The default
> `scan_map_inspect` workload is dominated by grid localization.

### The gRPC boundary is not the bottleneck

Measured directly by timing every `IngestStream.ingest_frame` call:

```
frames ingested         : 121
total run wall clock    : 3.582 s
gRPC round-trip total   : 0.210 s
gRPC share of whole run : 5.9%
  mean / p50 / p95 / max: 1.734 / 1.666 / 2.345 / 4.499 ms
```

> **Correction to the plan.** Phase 4 was justified partly on removing the
> synchronous per-tick round-trip to the Rust daemon. At **5.9% of the run**,
> removing it entirely is a minor win. The case for Phase 4 rests on the Python
> tick loop itself — localization, VIO, terrain sampling and numpy dispatch
> overhead — not on the IPC boundary. The boundary should be removed because it
> makes an in-process real-time viewer possible (Phase 5), not for throughput.

### Diffuse numpy overhead

Not attributable to one call site, but large in aggregate — these are
scalar/tiny-array operations where dispatch dominates the arithmetic:

| Operation | Cumulative | Calls |
|---|---:|---:|
| `numpy.ufunc.reduce` | 0.49 s (self) | 211,592 |
| `fromnumeric._wrapreduction` | 0.78 s | 189,102 |
| `np.min` + `np.max` | 0.83 s | 178,932 |
| `np.clip` | 0.75 s | 54,253 |
| `getlimits.finfo.__init__` | 0.12 s | 102,392 |

Roughly 700,000 numpy calls on 2- and 3-element arrays. This is the cost that
disappears wholesale in Rust regardless of which module is ported, and it is
spread across every module rather than concentrated anywhere fixable in Python.

## Cache effectiveness

From the deterministic counters now recorded in
`tests/golden/performance/*.json` (`intercept_stress`, 5 frames):

| Cache | Hits | Misses | Hit rate |
|---|---:|---:|---:|
| `terrain.height_at` | 1,275 | 11,766 | 9.8% |
| `visibility.los` | 8 | 185 | 4.1% |
| `obstacles.point_collides` | 0 | 11 | 0% |

The LOS cache quantizes ray endpoints to 1e-5 m. For continuously moving
platforms essentially every ray is unique, so the cache costs a hash and returns
nothing — as suspected. `terrain.height_at` fares only slightly better.

## What this implies for migration order

Ranked by measured benefit rather than by assumption:

1. **`GridLocalizer`** — 40% of the tick loop, self-contained, no dependency on
   the world crate. The single highest-value port, and it was not in the plan's
   first tranche.
2. **Terrain construction** (`_build_pyramid`, `build_land_cover_layer`) — 42% of
   a short run and directly responsible for the delay after pressing Run. Lands
   naturally with `argusnet-world`.
3. **VIO / EKF** — 13% of the tick loop, dense small-matrix work that suits Rust.
4. **Terrain sampling** (`height_at`) — 9%, and it is the shared leaf that other
   ports depend on, so it comes with `argusnet-world` anyway.
5. **LOS / visibility** — hot only under `--occlusion-aware-mapping`. Worth
   porting for that path, but it should not lead the ordering.
6. **Obstacle primitives** — a construction cost, not a tick cost. Port with the
   world crate for correctness and single-authority reasons, not for speed.

The gRPC boundary is not on this list.

## Reproducing

The deterministic half of the regression gate runs everywhere:

```bash
python -m pytest tests/test_performance_regression.py -q
```

Timing comparison is opt-in, because absolute baselines do not transfer between
machines:

```bash
ARGUSNET_RUN_PERF_TIMING=1 python -m pytest tests/test_performance_regression.py -q
```

Regenerate baselines after an intentional change:

```bash
ARGUSNET_UPDATE_PERF_GOLDEN=1 python -m pytest tests/test_performance_regression.py -q
```
