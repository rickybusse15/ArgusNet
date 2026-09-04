# ADR-005: Rust as the Runtime Authority

**Status:** accepted
**Date:** 2026-09-02
**Supersedes:** Decision 3 of `docs/ARCHITECTURE_DECISIONS.md`

## Context

`docs/ARCHITECTURE_DECISIONS.md` Decision 3 formalised the state of things in
March 2026: Python `TerrainLayer` is the authority for altitude clamping and LOS,
the viewer reads a pre-baked `viewer_mesh`, and the Rust `terrain-engine` crate is
"reserved for Rust-side analytic consumers". Decision 1 made `argusnet-core` the
sole producer of fused object state, but left everything else in Python.

Three years of that boundary has produced duplication rather than separation:

- **Terrain has three implementations** — Python `TerrainModel` (analytic),
  Python `TerrainLayer` (tiled heightmap), and Rust `terrain-engine::GridTerrain`
  — plus a fourth height sampler, `TerrainViewerMesh::sample_height`, inside the
  viewer.
- **Safety has two** — `rust/safety-engine` and `src/argusnet/safety/checker.py`,
  whose docstring says "The defaults mirror `rust/safety-engine/src/limits.rs`".
  The Python one runs; the Rust one is imported by nothing but its own bench.
- `terrain-engine` and `safety-engine` together are ~2,100 lines and 30 tests
  that **no shipping code depends on**, while `docs/architecture.md` lists them as
  real subsystem crates.

Separately, the viewer cannot become a real-time application under this boundary.
It shells out to `python3 -m argusnet sim` — probing hardcoded conda paths to find
an interpreter — waits for a multi-megabyte replay JSON, shells out again to bake
GLBs, then scrubs pre-computed frames. Nothing simulates while the operator
watches.

## Decision

**Rust becomes the authority for the runtime; Python is retained for offline
analysis.**

Specifically:

1. The per-tick runtime — world, terrain, sensing, mapping, planning, mission
   loop — moves to Rust crates. `argusnet-sim` owns the tick.
2. `terrain-engine` grows into `argusnet-world` and becomes the single terrain,
   obstacle, land-cover and line-of-sight authority for the simulation, the safety
   engine, and the viewer. The Python implementations are removed, not mirrored.
3. `safety-engine` is wired in and `src/argusnet/safety/checker.py` is deleted.
   There is one set of `DronePhysicalLimits`.
4. The viewer links `argusnet-sim` as a library and steps the world in-process.
   The Python subprocess path is removed.
5. Python retains offline evaluation, export/GIS (`pyproj`, `tifffile`, `fiona`),
   benchmark sweeps, and reporting — work that is not on the frame budget and
   where the Python ecosystem is a genuine advantage.

This supersedes Decision 3, which named Python `TerrainLayer` authoritative and
reserved `terrain-engine` for hypothetical Rust consumers.

## Rationale, with measurements

The pre-migration profile (`docs/performance/baseline-profile-2026-09.md`)
corrects two assumptions that were made before measuring, and they matter for
how this decision is justified:

- **This is not primarily about the gRPC boundary.** The per-tick synchronous
  round-trip to the daemon was timed at 1.73 ms/frame — **5.9%** of a
  representative run. Removing it is a minor throughput win. It is removed
  because an in-process world is what makes a real-time viewer possible, not for
  speed.
- **The cost is spread through the Python tick loop, not concentrated in one
  leaf.** `GridLocalizer.update` is 40% of the tick loop; VIO is 13%; terrain
  sampling 9%. Beneath all of it sit roughly 700,000 numpy calls on 2- and
  3-element arrays, where dispatch dominates arithmetic. That last cost
  disappears in Rust regardless of which module is ported and cannot be fixed in
  Python.

Single authority is the other half of the case, and it is independent of
performance: a duplicated implementation that is "kept in sync" by comment is a
defect generator. Two of the four terrain implementations already disagree —
`GridTerrain` has no LOD tiles, no vegetation transmittance, and no land-cover
attenuation.

## Consequences

### Positive
- One implementation per concern; the "mirror this constant" comments go away.
- The viewer can run the world at frame rate with no Python process involved.
- `terrain-engine` and `safety-engine` stop being dead weight that CI compiles,
  tests, lints and benchmarks on every run.
- The ~48-module `mypy` `ignore_errors` list is almost exactly the migration set.

### Negative
- Large, and touches the two biggest files in the repository.
- Determinism cannot be preserved bit-for-bit — see ADR-006.
- Rust iteration is slower than Python for scenario experimentation, which is
  part of why the evaluation and scenario-sweep layer stays in Python.
- Contributors need Rust to change simulation behaviour.

### Migration
Sequenced in the migration plan. Ordering follows the measured profile rather
than assumption: `GridLocalizer` first, then terrain construction, then VIO, then
terrain sampling. Obstacle primitives move for single-authority reasons, not
speed — 39,744 of 39,748 `_point_in_polygon` calls come from scenario
construction, not the tick loop.

## Affected Modules

| Module | Change type |
|--------|------------|
| `rust/terrain-engine` → `argusnet-world` | expanded, becomes load-bearing |
| `rust/safety-engine` | wired in, becomes load-bearing |
| `rust/argusnet-sim`, `-sensing`, `-mapping`, `-planning`, `-types` | new |
| `src/argusnet/world/`, `simulation/`, `sensing/`, `mapping/`, `localization/`, `planning/`, `safety/` | removed after port |
| `src/argusnet/evaluation/`, `core/config.py`, `cli/` | retained |
| `rust/argusnet-viewer` | links `argusnet-sim`; subprocess path removed |

## Tests Required

- Rust ports of `test_terrain_features.py`, `test_collision.py`,
  `test_visibility_probabilistic.py`.
- `tests/test_replay_contract.py` and `replay.rs::contract_fixture_tests` must
  stay green as the writer moves to Rust.
- The deterministic work-counter gate in `tests/test_performance_regression.py`,
  re-baselined per ADR-006 at each port.
- `tests/test_runtime.py::RustRuntimeParityTest` and
  `rust/argusnet-server/tests/grpc_parity.rs`.

## References

- `docs/performance/baseline-profile-2026-09.md`
- `docs/ARCHITECTURE_DECISIONS.md` (Decisions 1 and 3)
- `docs/KNOWN_GAPS.md` — "Rust world/terrain runtime integration: Partial"
- ADR-006 (determinism re-baseline)
