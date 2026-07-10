# Changelog

## Unreleased — Obstacle-Aware Mapping And Redirects

- Sim: new opt-in `--occlusion-aware-mapping` gates scan coverage and the split-view reconstruction
  through `EnvironmentQuery.los()`, so cells hidden behind buildings/walls are no longer stamped as
  scanned or populated with ground-truth height.
- Sim: with the same flag, adaptive/cooperative frontier redirects, frontier exploration, POI
  inspection redirects, and egress return-home trajectories use `PathPlanner2D` routes around hard
  blockers, falling back to the legacy straight-line hover trajectory when no route is available.
- Tests: added truth-isolation coverage for blocked reconstruction cells, obstacle-routed redirect
  geometry, collision-free sampled drone positions, and fixed-seed determinism for the opt-in mode.
- Perf: occlusion-aware coverage caps per-footprint LOS evaluations (`_COVERAGE_LOS_MAX_SAMPLES`,
  deterministic striding via `los_max_samples` on `CoverageMap.mark_circular` /
  `WorldMap.add_scan_observation`). Typical resolutions test every cell exactly; the cap only bites at
  very fine `--coverage-resolution-m`, bounding worst-case cost. Marked cells are still genuinely
  visible (invariant preserved).
- Tests: showcase test confirming occlusion scales with obstacle height/density — `military_compound`
  (tall buildings + walls) shadows ~4–5× more ground than open `coastal` terrain; near-nadir mapping
  over flat/undulating terrain is barely occluded (expected).

## Unreleased — Adaptive Cooperative Search And Headless View Modes

- Sim: `--adaptive-search` refines `--cooperative-search` — drones redirect via wedge-aware,
  coverage-balanced frontier selection (`select_cooperative_wedge_cell`), reallocating from finished
  wedges to the most under-covered one, and the wedge origin re-anchors on the fused team estimate
  (a localization→planning feedback loop). Measured vs plain cooperative on a medium map: coverage
  92% vs 81%, more even per-wedge balance, tighter team fix.
- Viewer: `live-stream` is now a default cargo feature, so a plain `cargo build --workspace` keeps
  the viewer's `--live` flag (build `--no-default-features` for a lean replay-only viewer).
- Viewer (headless): `render_headless` honors `--view-mode` — scan-map/split stills and PNG
  sequences now draw the accumulated reconstruction (height-colored) and the team co-localization
  beacon with spokes, and a single `--output` still renders the final frame. Makes the reconstruction
  and cooperative results CI-renderable (covered by a new headless test).

## Unreleased — Team Co-Localization In The Viewer

- Replay: cooperative runs now record the fused team position per frame as the additive
  `scan_mission_state.team_localization` (position, 1-sigma std, confidence, contributing drone IDs);
  documented in `docs/replay-schema.json` and modelled by `argusnet.core.types.TeamLocalization`.
- Viewer: renders the team estimate as a cyan beacon (stalk + sphere + true uncertainty ring) with
  spokes to each drone whose estimate was fused in, so cooperative co-localization is visible.
  Fixed the per-drone localization ellipses, which had used a stale `(x, z, -y)` mapping and drew in
  the wrong place — now Z-up `(x, y, z)` like every other overlay.

## Unreleased — Cooperative Radial Search

- Sim: new opt-in `--cooperative-search` (scan_map_inspect mode) launches every drone from one
  grounded origin and fans them out into disjoint angular wedges of the search area — a smarter,
  datum-anchored coverage pattern than independent lawnmower sectors. `--search-origin X,Y` sets the
  common origin (defaults to the search-area center). `cooperative_wedge_waypoints()` builds each
  drone's expanding polar boustrophedon.
- Sim: in cooperative mode the drones' independent per-drone localizations (all referenced to the
  shared coverage map they build together) are inverse-variance fused into a team estimate via the
  now-wired `GridLocalizer.fuse_estimates()`. The grounded origin, and the team co-localization
  convergence, are reported in the CLI summary and recorded additively in `meta.scenario_options`
  (`cooperative_search`, `search_origin_m`) and the run summary.

## Unreleased — Independent Split-View Camera Controls

- Viewer: mouse input is now routed to the region under the cursor. The reconstruction pane has its
  own `OrbitCamera`, so in split view the right pane can be orbited, panned, and zoomed independently
  of the left real-terrain view. Scroll and drag over the drawer/tabs are left to egui, so the panel
  scrolls without moving the cameras — each region (left view, reconstruction, drawer) has its own
  scroll wheel.

## Unreleased — Split-View 3-D Reconstruction And Coverage Resolution

- Sim: new `--coverage-resolution-m` flag sets the scan/reconstruction grid cell size (default 50 m,
  previously hard-coded), so mapping runs can produce a higher-resolution coverage map and denser
  reconstruction. Recorded additively as `meta.scenario_options.coverage_resolution_m`.
- Viewer: Split mode is now a genuine side-by-side — the right half is an empty area that fills with
  a 3-D reconstruction of the believed terrain, rendered by an angled perspective camera on a
  dedicated render layer (real terrain, drones, and gizmos no longer leak into it) over its own dark
  clear color. Reconstruction tiles auto-size to the coverage cell spacing so the surface reads as
  near-continuous at any resolution.
- Docs: `docs/usage.md` mapping recipe updated for a large, high-resolution path-planning run and the
  3-D split view.

## Unreleased — Viewer Split-View And Startup Options

- Viewer: `argusnet-viewer` gained `--view-mode {real-world,scan-map,split}` to choose the initial
  render (split shows real terrain on the left and the accumulated scan-map reconstruction on the
  right; the `M` key still cycles the modes at runtime) and `--autoplay` to start the replay
  timeline immediately so the reconstruction fills in without pressing Space. `ViewMode` is now
  re-exported from the crate root.
- Docs: `docs/usage.md` adds a mapping-run recipe that packages a scan/map/inspect mission and opens
  it directly in split view.

## Unreleased — Front-Door Reporting

- CLI: `argusnet sim` no longer prints the misleading "No fused tracks were produced." for
  mission modes where tracks are not the point. A `scan_map_inspect` run now reports final phase,
  map coverage, and POIs inspected; a `target_tracking` run that produced no tracks reports the
  observation acceptance rate and dominant rejection reason, with an actionable hint (e.g. use a
  smaller `--map-preset`) when targets are `out_of_range`. Report-only change — replay output and
  the fixed-seed determinism hash are unchanged.
- Docs: `docs/usage.md` documents that the default sim is a mapping mission and adds a curated
  target-tracking demo command that reliably produces fused tracks.
- Live: `argusnet live` gained a `--map-preset` flag (default `medium`) and passes it to the
  continuous sim, so the streamed target-tracking run keeps drones within sensor range of their
  targets and the viewer shows fused tracks instead of a nearly empty scene.
- Fixed: the bundled `scene.smartscene` manifest referenced `overlays/walls.glb`, which is not in
  the package; the dangling overlay layer was removed so the viewer no longer logs a missing-asset
  error on every load (`validate-scene` still reports the separately-missing replay payload).

## Unreleased — Usability, Realism, And Robustness Pass

- CLI: live progress bars for `sim` and `benchmark` runs (TTY-only, `--quiet`-aware); export
  formats listed in `--help`; `--dry-run` now fails fast on bad config files and warns about
  unrecognized config keys (`unknown_config_keys` in `core/config.py`).
- Performance: the simulation loop now ingests frames over the bidirectional `TrackStream` RPC
  (`TrackingService.open_ingest_stream()`) instead of per-frame unary calls; coverage-map marking
  is vectorized; per-pair RNG construction and sensor look-direction evaluation are hoisted out of
  the observation hot loop. Fixed-seed replay output is bit-identical to the previous behavior.
- Realism (opt-in flags, default off): `--enforce-flight-envelope` clamps drone motion to
  `FlightEnvelope` acceleration, bank-derived turn-radius, and climb/descent limits
  (`EnvelopeLimitedTrajectory`); `--frontier-exploration` drives scanning drones with
  `FrontierPlanner.select_frontier_cell()` instead of fixed lawnmower sectors.
- Realism (additive): `wind_drift_scale` exposed in `DynamicsConfig`/`dump-config`; wind- and
  climb-aware return-home battery reserve (`BatteryModel.dynamic_reserve_wh`, static reserve as
  floor); optional GNSS Gauss-Markov colored noise and deterministic outage windows
  (`GNSSSimulator`, `OutageSchedule`).
- Robustness: Rust tracker reconditions degenerate innovation covariances (symmetrize + jitter
  retry) instead of erroring, bounds covariance growth during pathological coasts, and gains a
  multi-track long-duration stress test; collision push loops log a warning when they exhaust
  their iteration budget.
- Fixed: `MissionExecutor` inspect-task completion crashed any run reaching the inspect phase
  (`all_complete` property was called as a method).

## Unreleased — Benchmarking And Performance Harness

- Added the `argusnet benchmark` CLI, canonical fast/slow scenario suites, performance summaries,
  and fast-suite golden regression checks.
- Added Python hot-path benchmark tests, Rust Criterion benchmark compile coverage, and nightly
  benchmark workflow artifact upload.
- Added terrain, gradient, LOS, and obstacle cache metrics plus per-frame simulation timing fields.

## 2026-04-29 — Documentation Reframing

- Reframed documentation around ArgusNet as a map, localize, inspect, revisit, and coordination
  system.
- Updated architecture and state ownership docs to use `src/argusnet` and `argusnet-*` crate names.
- Clarified current `scan_map_inspect` behavior: mapping coverage, localization confidence,
  inspection POI completion, deconfliction, egress, and replay state.
- Replaced outdated scenario and planning vocabulary with mapping, localization, inspection,
  revisit, POI assignment, and multi-drone coordination terminology.

## 2026-04-27 — Scan, Localize, Inspect, Egress

- Added scan → localize → inspect → egress mission state.
- Added scan coverage thresholds, POI localization ellipses, and inspection event logging.
- Added egress/RTH with per-drone home progress and arrival detection.
- Hardened multi-drone deconfliction.
- Improved viewer panels, selection details, event display, and scan overlay color.

## March 2026 — Viewer, Scene, Terrain, And Ingestion Foundations

- Consolidated documentation into `README.md` and `docs/`.
- Added native Bevy viewer and `smartscene-v1` package loading.
- Split scene compilation into terrain, geometry, GIS, and styling helpers.
- Added terrain presets, terrain-aware mission zones, rejection diagnostics, and export formats.
- Added MQTT and file replay ingestion adapters.
- Added Rust fusion service, replay export, and Python simulation workflow.
