# Performance and Benchmarking Standard

This document defines the ArgusNet efficiency, benchmarking, data-layout, caching, language-boundary, and code-quality standards. It is written as an implementation guide for both human engineers and AI coding agents.

ArgusNet is a closed-loop belief-world mission system, not only a replay viewer. Performance work must therefore preserve safety, determinism, state ownership, and architectural boundaries before optimizing raw speed.

---

## 1. Goals

ArgusNet performance work has four goals:

1. **Make every change measurable.** Runtime, memory, latency, determinism, and mission-quality metrics must be benchmarked before and after nontrivial changes.
2. **Protect closed-loop correctness.** Optimization must not bypass localization confidence, belief-world authority, mission constraints, or safety validation.
3. **Keep hot paths predictable.** Per-frame simulation, sensor fusion, terrain queries, planning, replay generation, and viewer playback must avoid unbounded allocations and hidden global state.
4. **Choose the right language for the job.** Rust owns latency-sensitive runtime math and services. Python remains the orchestration, scenario, analysis, export, and experimentation layer unless a measured bottleneck justifies migration.

---

## 2. Current Baseline Assumptions

The current implementation is mixed Python and Rust:

- Python handles simulation setup, scenario generation, environment/terrain modeling, replay/export tooling, and CLI orchestration.
- Rust handles authoritative sensor-fusion math, the gRPC runtime daemon, protobuf conversion, and the Bevy viewer.
- Existing docs already define benchmark scenario families, replay evaluation metrics, terrain cache expectations, and code-completion standards.

This document makes those standards operational by defining when to benchmark, what to measure, when to migrate code between languages, and how data structures should be shaped for future modules.

---

## 3. Benchmark Levels

Every performance claim should identify which benchmark level it uses.

### Level 0: Microbenchmark

Use for isolated hot functions.

Examples:

- Terrain `height_at`, `gradient_at`, `los_raycast`, and `comms_shadow`
- Obstacle containment and segment intersection
- Coordinate transforms such as WGS84/ECEF/ENU conversions
- Rust Kalman predict/update, association, and gating
- Serialization/deserialization for protobuf and replay structures
- Planner cost functions and nearest-neighbor searches

Required tools:

- Rust: `criterion`, `cargo bench`, `iai-callgrind` where deterministic instruction counts matter
- Python: `pytest-benchmark`, `time.perf_counter_ns`, `py-spy`, `scalene`, `cProfile` for coarse profiling

Output requirement:

- Median, p95, and p99 latency
- Allocation count or estimated memory churn when practical
- Input size and seed
- Commit SHA and benchmark command

### Level 1: Module Benchmark

Use for an entire subsystem running on synthetic but realistic data.

Examples:

- Terrain query batch over a map tile
- Sensor observation generation for `N` drones and `M` observed objects or POIs
- Fusion service ingest for a full frame
- Planner route generation across obstacle-dense scenes
- Replay writer/exporter processing a mission log
- Viewer loading a `.smartscene` package

Required output:

- Throughput, for example frames/sec, queries/sec, observations/sec, routes/sec, or MB/sec
- Peak RSS memory
- Error/failure count
- Determinism check for seeded modules

### Level 2: Scenario Benchmark

Use for mission-level behavior.

Examples:

- `baseline_coverage`
- `intercept_stress`
- `persistent_long`
- `search_acquisition`
- `planner_adversarial`

Required output:

- All `EvaluationReport` metrics
- Wall-clock runtime
- CPU utilization when available
- Peak memory
- Replay size
- Number of frames, observations, fused object states, planning events, safety events, and exported artifacts

### Level 3: System Benchmark

Use before major releases or architectural migrations.

Examples:

- Full Python simulation + Rust fusion daemon + replay export + viewer load
- Long multi-seed benchmark sweep
- Large map and scene package ingestion
- Realistic closed-loop runtime test with mission execution, mapping, indexing, planning, and safety logging

Required output:

- Scenario benchmark metrics
- End-to-end latency budget by subsystem
- Resource budget by process
- Regression comparison against previous accepted baseline
- Reproducible environment manifest

---

## 4. Standard Benchmark Matrix

Every accepted performance baseline should record at least this matrix.

| Category | Metric | Target direction | Notes |
|---|---:|---|---|
| Runtime | wall_clock_s | lower | Total benchmark execution time |
| Runtime | frame_time_mean_ms | lower | Mean closed-loop frame time |
| Runtime | frame_time_p95_ms | lower | Primary smoothness/latency gate |
| Runtime | frame_time_p99_ms | lower | Catches rare stalls |
| Throughput | frames_per_sec | higher | Useful for replay/sim throughput |
| Throughput | observations_per_sec | higher | Sensor/fusion stress metric |
| Throughput | terrain_queries_per_sec | higher | Terrain hot-path metric |
| Memory | peak_rss_mb | lower | System and module level |
| Memory | replay_size_mb | lower | Replay/export storage pressure |
| Quality | mission_completion_rate | higher | Must not regress while optimizing |
| Quality | localisation_rmse_m | lower | Track/localization quality gate |
| Quality | fusion_continuity_mean | higher | Coverage/fusion quality gate |
| Reliability | safety_override_count | lower | Target is zero for normal scenarios |
| Reliability | infeasible_path_rejection_count | lower | Allowed in adversarial scenarios if logged |
| Determinism | repeated_seed_diff_count | zero | Same seed should produce identical outputs |

Performance gains are not acceptable if they break required mission metrics, deterministic tests, or safety gates.

---

## 5. Data Type and Schema Standards

### 5.1 Units and numeric precision

Use explicit units in field names or type wrappers.

Required naming examples:

- `x_m`, `y_m`, `z_m`
- `timestamp_s`
- `duration_s`
- `speed_mps`
- `heading_rad`
- `covariance_m2`
- `energy_reserve_fraction`

Default precision:

| Layer | Default numeric type | Reason |
|---|---|---|
| Rust runtime math | `f64` | Stable filtering, geometry, covariance, terrain, localization |
| Rust viewer GPU buffers | `f32` | Rendering and mesh buffers only |
| Python simulation math | `float` | Python float is double precision |
| Python vectorized arrays | `np.float64` for authoritative math, `np.float32` for visual/export-only arrays | Avoid accidental precision loss |
| Protobuf wire schema | `double` for authoritative state, `float` only for visual/large approximate arrays | Preserve cross-language state consistency |

Do not change an authoritative state field from double precision to single precision without a benchmark and accuracy comparison.

### 5.2 Identifiers

Use stable string IDs at external boundaries and compact integer IDs inside hot loops.

Recommended pattern:

- External/replay/protobuf: stable string IDs such as `drone_id`, `poi_id`, `object_id`, and `mission_id`
- Internal Rust hot path: map external IDs to `u32` or `usize` indexes through an ID table
- Python batch processing: use integer indexes in arrays, preserve a sidecar `id_to_index` / `index_to_id` mapping

This keeps logs readable while avoiding repeated string hashing in per-frame loops.

### 5.3 Geometry types

Use structured geometry types rather than loose tuples in public APIs.

Rust:

- Use small `Copy` structs for `Point2`, `Point3`, `Vector3`, `Pose3`, `Aabb2`, `Aabb3`, and `FrameId`.
- Use `nalgebra` or a clearly selected math crate consistently for linear algebra.
- Avoid exposing raw `[f64; 9]` covariance in new internal APIs; wrap it in a covariance type with checked construction.

Python:

- Use frozen dataclasses or Pydantic/msgspec models at API boundaries.
- Use NumPy arrays for dense numeric batches.
- Avoid dictionaries in hot loops unless they are static lookup maps.

### 5.4 State snapshots

Per-frame state should be append-only and replayable.

Rules:

- Mutable runtime state lives with the subsystem owner.
- Evaluation/replay data should be immutable snapshots or structured event logs.
- No module should silently mutate another module's authoritative state.
- State passed across async boundaries must be copy-safe, versioned, and timestamped.

### 5.5 Serialization format choices

| Use case | Preferred format | Reason |
|---|---|---|
| Runtime service boundary | Protobuf/gRPC | Stable typed contract |
| Human-readable small config | TOML/YAML | Easy editing |
| Large numeric arrays | Arrow, Parquet, NumPy `.npz`, or memory-mapped binary | Avoid giant JSON |
| Replay metadata | JSON with schema | Human-readable and compatible with current tooling |
| Heavy replay frames | JSONL, Arrow, or chunked binary sidecar when JSON becomes too large | Avoid single massive files |
| Viewer scene package | Existing `.smartscene` plus binary sidecars for large meshes/arrays | Keep package load fast |

Large replay, scene, or array files should not be committed directly. Use generated artifact directories, Git LFS only when the project intentionally stores a large reference artifact, or external release assets.

---

## 6. Error Handling and Data Handling Improvements

### 6.1 Typed errors

New modules should define typed errors instead of returning bare strings or generic exceptions.

Rust:

- Use `thiserror` for library errors.
- Use `anyhow` only at binary/application boundaries.
- Use `Result<T, E>` for recoverable failures.
- Reserve `panic!` for impossible programmer errors, not runtime data problems.

Python:

- Define subsystem-specific exception classes.
- Include subsystem, input ID, timestamp, and cause where possible.
- Do not swallow exceptions in benchmark or mission execution code unless the fallback is explicitly logged.

### 6.2 Stale and missing data

Every runtime data structure with time meaning must define:

- `timestamp_s` or `timestamp_utc`
- `source_id`
- `frame_id` where relevant
- staleness threshold
- fallback behavior

Examples:

- Stale localization should gate inspection and map-relative planning.
- Missing terrain should fall back conservatively, not assume clear space.
- Missing sensor data should degrade confidence and be visible in metrics.
- Missing comms should trigger mission execution fallback, not silent continuation.

### 6.3 Validation

Validate at subsystem boundaries:

- Unit ranges, for example probabilities in `[0, 1]`
- Finite floats, no NaN/Inf in runtime state
- Coordinate frame compatibility
- Monotonic timestamps per source stream
- Nonnegative covariance diagonal entries
- Geofence containment for candidate routes
- Replay schema compatibility

---

## 7. Caching Standard

Caching is encouraged only when ownership, invalidation, and measurement are clear.

### 7.1 Cache documentation requirement

Every cache must document:

- Owner module
- Key type
- Value type
- Maximum size
- Eviction policy
- Invalidation triggers
- Thread-safety model
- Metrics emitted, such as hits, misses, evictions, and rebuild time

### 7.2 Recommended caches

| Cache | Owner | Key | Value | Invalidation |
|---|---|---|---|---|
| Analytic terrain point cache | World/Terrain | quantized `(x_m, y_m, cell_m)` | height | new terrain model |
| Terrain gradient cache | World/Terrain | quantized `(x_m, y_m, delta_m)` | gradient | new terrain model |
| LOS segment cache | Visibility/Sensing | quantized origin/endpoint/clearance + scene version | blocker result | scene/terrain/obstacle update |
| Obstacle spatial index | World/Obstacles | scene version | R-tree/BVH/grid | obstacle set update |
| Planner route cache | Planning | start/goal/geofence/obstacle version/planner config | route candidate | map or constraints update |
| Map tile cache | Mapping/World | tile ID + LOD + version | belief/elevation/occupancy tile | tile update/version bump |
| Keyframe descriptor cache | Indexing | keyframe ID + descriptor version | descriptor/embedding | descriptor model update |
| Proto conversion cache | Service boundary | schema version + stable ID table | ID mappings/converters | schema or ID table update |
| Viewer mesh cache | UI/Viewer | scene version + LOD | GPU-ready mesh | scene reload |

### 7.3 Cache rules

- Never cache unsafe, unvalidated commands.
- Never let a cache hide state updates from safety validation.
- Cache keys must include scene/map/config version when geometry or constraints affect the result.
- Bounded caches are required in long-running services.
- Cache metrics should be visible in benchmark reports.

---

## 8. Language Boundary and Migration Guidance

### 8.1 Keep in Python

Python is appropriate for:

- CLI orchestration
- Scenario generation
- Research experiments and sweeps
- Data export/import tools
- Replay post-processing
- Plotting and analysis
- Lightweight config parsing
- Initial prototypes before the API stabilizes

Python code must still be deterministic, tested, typed where practical, and profiled before being blamed for performance issues.

### 8.2 Move to Rust when justified

Move a module or kernel to Rust when at least one condition is true:

- It is in the per-frame closed-loop path and consumes more than 10 percent of the frame budget.
- It performs heavy geometry, spatial indexing, filtering, path planning, or map updates.
- It must run inside a long-lived service with strict latency and memory bounds.
- It must be shared by the viewer, daemon, and simulation without Python dependency.
- It requires safe concurrency over large state.

Candidate Rust modules:

- `terrain-engine`: analytic terrain grids, interpolation, LOS raycasts, slope/curvature queries
- `world-engine`: obstacle spatial indexes, map bounds, visibility acceleration
- `mapping-engine`: occupancy/elevation/belief tile updates
- `localization-engine`: pose graph, relocalization, map-relative correction
- `planner-engine`: route/viewpoint planning, frontier search, next-best-view scoring
- `trajectory-engine`: dynamic feasibility and path smoothing
- `safety-engine`: geofence, collision, altitude, comms, and abort validation
- `indexing-engine`: spatial memory lookup and keyframe retrieval when query volume is high
- `eval-suite`: deterministic large scenario evaluation and metric aggregation

### 8.3 Use C++ only by exception

C++ is not the default for ArgusNet. Consider C++ only when integrating a mature robotics, geometry, or vision library that is already C++ native and has no strong Rust equivalent.

If C++ is introduced:

- Wrap it behind a Rust or Python boundary.
- Keep unsafe/native code small.
- Add ABI/build documentation.
- Add deterministic tests across supported platforms.
- Do not let C++ become an ungoverned parallel runtime.

### 8.4 GPU acceleration

Use GPU acceleration only after CPU baselines are measured.

Potential GPU paths:

- Viewer rendering, already Bevy/GPU oriented
- Dense map updates
- Large batch image/depth processing
- Large embedding/descriptor workloads

Do not use GPU kernels for core safety gates unless there is a CPU fallback and deterministic validation.

---

## 9. Module-by-Module Standards

### 9.1 World and Terrain

Standards:

- Analytic terrain is authoritative for planning, safety, sensing, and physics.
- Visual terrain is rendering-only.
- Terrain queries must expose explicit bounds and conservative out-of-bounds behavior.
- Add curvature and slope APIs before planners depend on ridge/valley reasoning.
- Use shared grids or memory-mapped arrays for Rust-side terrain when query rate is high.

Benchmark gates:

- point queries/sec
- LOS raycasts/sec
- p95 terrain query latency
- cache hit/miss ratio
- consistency against Python reference implementation

### 9.2 Sensing and Fusion

Standards:

- Rust remains authoritative for fused object states.
- Python may synthesize observations, but fusion lifecycle, covariance, and health state should stay in Rust.
- Observations must include source ID, timestamp, frame ID, covariance/noise model, and rejection reason when rejected.
- Association/gating changes require deterministic scenario comparisons.

Benchmark gates:

- ingest frame latency
- observations/sec
- association complexity versus number of drones and observed objects
- localization/fusion error metrics
- stale fused-state reacquisition metrics

### 9.3 Localization

Standards:

- Pose estimates must include frame ID, covariance/confidence, timestamp, and source fusion status.
- Map-relative planning and inspection must be gated by localization confidence.
- Relocalization after restart/battery swap must be explicit and testable.
- Coordinate transforms must be centralized and fuzz-tested.

Benchmark gates:

- pose update latency
- relocalization latency
- localization RMSE in seeded scenarios
- covariance consistency

### 9.4 Mapping

Standards:

- Belief-world state is the authority for physical-mode planning.
- Ground truth is allowed only for simulation observation generation and scoring.
- Tile updates should be versioned and append/audit friendly.
- Dense map arrays should use compact numeric storage, not nested Python objects.

Benchmark gates:

- tile update throughput
- map memory per square kilometer or per cell
- coverage update latency
- uncertainty reduction metric

### 9.5 Indexing

Standards:

- Keyframes, landmarks, evidence, and map tiles require stable IDs and timestamps.
- Retrieval APIs must state whether they are approximate or exact.
- Descriptor/embedding model versions must be stored with artifacts.
- Cross-mission retrieval must preserve coordinate frame lineage.

Benchmark gates:

- insert throughput
- query latency p95/p99
- index size
- recall/precision for seeded retrieval tests

### 9.6 Planning, Trajectory, and Safety

Standards:

- Mission intent must pass through candidate task, route/viewpoint, trajectory proposal, safety validation, and executable command.
- Planners may propose; safety validates.
- Planning must use belief-world and localization state, not hidden truth.
- Route caches must include map, obstacle, geofence, and config version.

Benchmark gates:

- route planning latency
- rejected plan count
- safety override count
- mission completion rate
- energy reserve

### 9.7 Mission Execution and Evaluation

Standards:

- Mission runtime should emit structured events for planning, safety, sensing, localization, mapping, and execution transitions.
- Replays should be schema-versioned and additive.
- Benchmark reports must include environment, command, seed, commit SHA, and scenario name.
- Slow scenario suites should run nightly or before release; fast suites should run on every PR.

Benchmark gates:

- end-to-end frame time
- replay write throughput
- evaluation report generation time
- deterministic repeat checks

### 9.8 Viewer and UI

Standards:

- Viewer meshes are display artifacts, not planning authority.
- Viewer should not call heavy analytic terrain APIs during rendering.
- Large scene loading should be async/background with progress reporting.
- Viewer should display benchmark and safety events without changing runtime state.

Benchmark gates:

- scene load time
- replay playback FPS
- memory usage for large scenes
- UI interaction latency

---

## 10. Tooling Standard

### 10.1 Required local commands

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python3 -m pytest tests/ -q
python3 -m ruff check .
python3 -m mypy src tests
```

If a tool is not yet configured, add the config before requiring it in CI.

### 10.2 Benchmark commands

Recommended command shape:

```bash
cargo bench --workspace
python3 -m pytest tests/ -q -m benchmark_fast
python3 -m pytest tests/ -q --benchmark-only
argusnet sim --duration-s 60 --seed 7
argusnet benchmark --suite fast --seeds 7 --output runs/benchmarks/<timestamp>
```

Current evaluation helpers live under `argusnet.evaluation.metrics`, `argusnet.evaluation.reports`,
`argusnet.evaluation.benchmarks`, and `argusnet.evaluation.scenarios`. The shipping benchmark CLI
runs canonical scenario suites, writes one `performance_summary.json` per scenario/seed, and writes
an aggregate `suite_summary.json` under the requested output directory.

```bash
argusnet benchmark \
  --suite fast \
  --seeds 7,42,137,9999,31415 \
  --output runs/benchmarks/<timestamp>
```

### 10.3 Profiling tools

| Area | Tool |
|---|---|
| Rust CPU profiling | `cargo flamegraph`, `perf`, `samply` |
| Rust microbenchmarks | `criterion`, `iai-callgrind` |
| Rust memory | `heaptrack`, `dhat`, `valgrind massif` where available |
| Python CPU | `py-spy`, `scalene`, `cProfile` |
| Python memory | `tracemalloc`, `memray`, `scalene` |
| Cross-process tracing | OpenTelemetry spans or structured JSON logs |
| Experiment logging | MLflow-compatible run directories |
| Large arrays/tables | Arrow/Parquet/NumPy memory maps |

---

## 11. CI and Regression Policy

### 11.1 Pull request gates

Every PR should run:

- Rust format/lint/test
- Python lint/type/test
- replay schema tests
- deterministic seed test
- fast benchmark smoke test when relevant code changes

### 11.2 Nightly or release gates

Nightly/release runs should include:

- slow benchmark scenarios
- multi-seed scenario sweeps
- memory profiling on representative large scenes
- replay/viewer load regression test
- benchmark report artifact upload

### 11.3 Golden performance files

This is a standard for accepted baselines. The repository may not yet contain these files for every
scenario; add them when a benchmark is promoted to a regression gate.

Store accepted benchmark summaries under:

```text
tests/golden/performance/
  benchmark_fast_seed7.json
  benchmark_fast_seed42.json
  scenario_<name>_summary.json
```

Golden files should include:

```json
{
  "schema_version": "argusnet-performance-v1",
  "commit_sha": "...",
  "created_at_utc": "...",
  "command": "...",
  "environment": {
    "os": "...",
    "cpu": "...",
    "memory_gb": 0,
    "rustc": "...",
    "python": "..."
  },
  "metrics": {
    "wall_clock_s": 0,
    "frame_time_p95_ms": 0,
    "peak_rss_mb": 0,
    "mission_completion_rate": 0
  }
}
```

---

## 12. Regression Thresholds

Use three levels of performance regression.

| Level | Change | Required action |
|---|---:|---|
| Green | within ±5 percent | Accept if quality/safety metrics pass |
| Yellow | 5 to 20 percent slower or larger | Explain in PR and confirm tradeoff |
| Red | more than 20 percent slower or larger | Block unless an ADR approves the tradeoff |

A performance improvement is also blocked if it causes any of the following:

- nondeterministic outputs for fixed seeds
- degraded mission pass/fail status
- bypassed safety validation
- hidden ground-truth use in physical-mode planning
- unbounded memory growth in long-running services
- dropped replay/evaluation visibility

---

## 13. Coding Standards for Future and Past Modules

### 13.1 Existing modules

When modifying existing Python or Rust modules:

- Preserve public interfaces unless an ADR approves a breaking change.
- Add compatibility wrappers when moving modules.
- Keep replay schema changes additive.
- Add tests before refactoring hot paths.
- Write migration notes when moving authority from Python to Rust.

### 13.2 New modules

New modules must include:

- documented owner subsystem
- state authority statement
- input/output schemas
- typed errors
- deterministic test fixture
- benchmark fixture if it touches a hot path
- logging/metrics hooks
- cache strategy if it caches anything

### 13.3 Rust standards

- Use `cargo fmt` and `clippy` clean code.
- Prefer `Result` over panics.
- Avoid `unsafe`; document and test if unavoidable.
- Use traits for subsystem boundaries.
- Keep data structures cache-friendly in hot loops.
- Avoid allocation inside per-frame loops.
- Use `Arc`/channels deliberately; document ownership and threading.

### 13.4 Python standards

- Use type hints for public functions.
- Prefer dataclasses or Pydantic/msgspec models at boundaries.
- Use NumPy for dense math and batch operations.
- Avoid large nested dictionaries/lists for numeric frame data.
- Keep random seeds explicit.
- Keep CLI tools thin; put testable logic in modules.

### 13.5 Documentation standards

Docs should answer:

- What subsystem owns this?
- What state is authoritative?
- What units and frames are used?
- What invariants must not break?
- What benchmark proves this works?
- What happens when data is missing, stale, or invalid?

---

## 14. Implementation Roadmap

### Phase 1: Baseline and reporting

1. Add a benchmark runner that executes fast scenario families with fixed seeds.
2. Emit `performance_summary.json` beside each evaluation report.
3. Add environment capture: commit SHA, OS, Python, Rust, CPU, memory.
4. Add p95/p99 frame timing to replay/evaluation metadata.

### Phase 2: Hot-path profiling

1. Profile terrain queries, LOS, obstacle checks, observation generation, gRPC conversion, and Rust fusion ingest.
2. Add microbenchmarks for the top five hot functions.
3. Add cache metrics for terrain and visibility.
4. Replace giant JSON numeric arrays with chunked or binary sidecars where needed.

### Phase 3: Data layout cleanup

1. Introduce typed geometry and covariance wrappers.
2. Add stable integer ID tables for hot loops.
3. Normalize units and frame IDs across replay, proto, and internal APIs.
4. Add validation for finite floats, covariance, timestamps, and frame compatibility.

### Phase 4: Rust migration candidates

1. Move terrain grid queries and LOS acceleration to `terrain-engine` or `world-engine` after profiling confirms benefit.
2. Move obstacle spatial indexing to Rust if planner/visibility benchmarks show pressure.
3. Move mapping tile updates to Rust when dense belief-world updates become per-frame.
4. Move planner kernels to Rust only after Python planner APIs stabilize.

### Phase 5: CI enforcement

1. Add fast benchmark smoke test to PR CI.
2. Add nightly slow benchmark workflow.
3. Store benchmark artifacts.
4. Fail red regressions unless an ADR approves the tradeoff.

---

## 15. Definition of Done for Performance Work

A performance change is complete only when:

- The bottleneck is measured before the change.
- The new behavior is benchmarked after the change.
- Safety, determinism, replay, and evaluation tests still pass.
- Cache invalidation is documented if caching was added.
- Data type changes include accuracy comparisons.
- Language migration includes compatibility boundaries and rollback strategy.
- Documentation is updated.

---

## 16. Related Documents

Read these with this standard:

- `docs/architecture.md` for subsystem boundaries and state ownership summary
- `docs/STATE_OWNERSHIP.md` for authoritative runtime state
- `docs/SCENARIOS.md` for benchmark scenario families and evaluation metrics
- `docs/TERRAIN.md` for analytic terrain interface and terrain caching
- `docs/PLANNING.md` for planner-to-trajectory expectations
- `docs/SAFETY.md` for physical/safety constraints
- `CONTRIBUTING.md` for code completion standards and definition of done
- `docs/adr/000-template.md` for architecture decision records
