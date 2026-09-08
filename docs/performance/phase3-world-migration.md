# Phase 3: shared Rust world authority

Base: GitHub `main` at `5f86094`. This change implements ADR-005's world authority
and uses ADR-006's explicit re-baseline policy for a verified visibility fix.

## Measurement

Measured sequentially on the same macOS arm64 machine, Python 3.12.13, with the
same Rust tracking daemon. The migrated world library uses the release profile.
Raw measurements and configuration are in `phase3-world-migration.json`.

| Work | Main before migration | Rust authority | Ratio |
| --- | ---: | ---: | ---: |
| 10,000 analytic heights, alpine, median of 5 | 197.70 ms | 0.915 ms | 216x faster |
| 257x257 tiled terrain construction, median of 5 | 42.80 ms | 24.47 ms | 1.75x faster |
| Representative scenario construction | 263.80 ms | 259.25 ms | effectively unchanged |
| Mean simulation tick | 4.00 ms | 3.94 ms | effectively unchanged |
| p95 simulation tick | 9.86 ms | 9.39 ms | small change |

The representative run is `medium`, `scan_map_inspect`, 6 drones, seed 7,
10 seconds at dt=0.25 (41 frames including time zero). These short local runs
are observations, not a portable timing gate or a claim about sustained viewer
frame rate. The original Phase 0 profile remains the migration-priority source:
localization and VIO still dominate important workloads. Scenario construction
also retains Python procedural shaping and metadata conversion costs.

The native bridge uses owned handles for persistent world objects and contiguous
f64 input/output arrays for height and differential batches. JSON is restricted
to object construction and smaller control/query messages; full terrain grids
are not re-serialized on every tick.

## Why the deterministic work baselines changed

Compared the benchmark replay output against an isolated copy of `5f86094`,
using the same interpreter and tracking daemon:

- `baseline_coverage`: observation counts unchanged (0 accepted, 0 rejected).
- `search_acquisition`: observation counts unchanged (16 accepted, 4 rejected).
- `intercept_stress`: accepted 16 -> 8; tracker rejections 5 -> 7. The Rust DDA
  correctly finds terrain occlusion from `ground-hotel` toward `asset-hawk`
  and `asset-orca`. Python's negative-direction denominator skipped cells.
  Fewer generated bearings also leave additional clusters with insufficient
  observations. Node states match exactly; inspected truth-state differences
  are floating-point rounding (about 3e-14 m), not changed trajectories.
- `intercept_stress` terrain cache hits/misses: 1276/11766 -> 94/338.
- `search_acquisition` terrain cache hits/misses: 385/8513 -> 73/326.

The cache reductions are intentional: LOS and differential queries now evaluate
inside Rust instead of making thousands of Python cached scalar calls. These
counters measure the Python query boundary, not every native interpolation.
The strict work-counter gate is retained and the new golden outputs are checked
in; it is not disabled or widened. Timing reference metadata is refreshed for
this machine, and cross-machine timing comparisons remain opt-in.

Regression tests pin reverse-direction DDA traversal, ridge occlusion, clipped
rays, LOD seams and missing tiles. A frozen fixture generated from `5f86094`
checks all 11 analytic presets in Rust and Python to 1e-9 m absolute tolerance.

## Ownership and scope

- `argusnet-world` owns terrain evaluation, tiled construction and mip pyramids,
  obstacle geometry/indexing, land-cover queries, visibility and detection.
- `argusnet-native` owns the Python ABI and native object lifetimes. Source
  installs build the library; wheels carry it. There is no runtime auto-build
  and no Python fallback.
- `safety-engine::simulation` owns the existing simulation safety policy, using
  the crate's physical-limit defaults. It accepts the telemetry Python actually
  has; it does not invent RSSI or gimbal signals to satisfy the richer flight
  validator. The opt-in gate and simulation clamp ordering stay in place.
- The viewer's replay raster adapter lives in `argusnet-world` and shares the
  interpolation kernel. Replay metadata is unchanged.
- Python retains configuration/preset data, GIS/import/export, procedural scene
  shaping and the simulation loop. Removing those belongs to later phases.

World bindings hold snapshots. Updating a world means replacing layers through
`EnvironmentModel`, which also invalidates scene-versioned LOS caches. Exported
NumPy arrays are not a second mutable runtime authority.

## Verification

- Rust workspace build and tests; 192 tests passed, including live gRPC parity.
- Replay-only viewer: 93 tests passed with `--no-default-features`.
- Workspace Clippy with `--all-targets -- -D warnings`, Rust formatting,
  Python ruff lint/format and mypy passed.
- Python full suite: 940 passed, 10 skipped (plus 65 subtests). The final native
  safety/runtime/replay/occlusion boundary suite also passed. The timing-only
  perf gate remains skipped unless calibrated.
- `cargo bench --workspace` completed. Dedicated tiled-world benchmarks measured
  4096 tiled height queries at 26.59 us and one two-level DDA ray at 0.683 us
  (20 samples, 1-second warmup and measurement windows). These are query timings,
  not a rendered frame budget.
- The headless Bevy render produced visible terrain and scene geometry at 640x480
  using the isometric preset. GPU access required running outside the sandbox.
- An editable install and a built wheel both load the native library. The wheel
  was exercised outside the source tree, including safety and replay-schema loading.

The viewer currently has no automated steady-state frame-time budget bench in
this base checkout. The headless still verifies rendering, not sustained 60 fps;
that remains part of the later real-time viewer work.
