# P4.9a.2: an `argusnet-rs` replay carries what the viewer shows

**Status:** delivered 2026-09-23, locally verified only (not pushed). Package ID
`P4.9a.2`. See [STATUS.md](STATUS.md).

Before this slice, P4.9a's `argusnet-rs` wrote a replay the schema accepted
but the viewer could not present. Frames had nodes, truths, observations,
tracks and metrics, and nothing else. The panels had no coverage, zones,
mission phase, POIs, deconfliction or generation rejections to show. `meta`
had no terrain, so there was no world to draw.

Now a replay from `argusnet-rs` carries every payload a Python replay of the
same bundle carries. Each one is either proven equal to the reference's own
serialization, or proven to differ only where stage 6's draws reach
(ADR-006).

## What a frame carries now

| Payload | Built from | Against the reference |
| --- | --- | --- |
| `generation_rejections` | stage 6: every rejected pair, with the reference's detail text, origin, attempted point, and the sight line's blocker, first hit and closest point | exact wherever both runs rejected a pair at the same gate |
| `mapping_state` | stage 8's coverage grid, now assembled in **every** mission mode | exact |
| `inspection_events` | stage 8: a coverage map per mission zone, and which drones are inside it | exact |
| `deconfliction_events` | stage 3's resolutions | exact |
| `localization_state` | the frame's track covariances and the tick's generated observations, through numpy's pairwise mean | draw-dependent; present exactly when there are tracks |
| `target_metadata` | truth order, the bundle's target behaviours, stage 6's per-target tally | exact except `status` and `confidence`, which follow accepted observations |
| `scan_mission_state` | the mission machine, stage 9's POI report and coordinator, stage 8's sticky estimates, newly covered cells and team fusion | exact except the localizer's estimate values, whose drone and timestamp match |

A payload the writer does not produce is omitted, not written empty. That
covers `belief_cells`, `pose_estimates`, `evidence_records`,
`platform_health` and `mission_events`, which Python writes empty in these
scenarios, and `safety_events` in a run without blocking safety. Since P4.9d,
a `safety_blocking` run writes stage 4's events
([SAFETY_AND_PROFILES.md](SAFETY_AND_PROFILES.md)). It also covers an empty event list, a scan state
outside `scan_map_inspect`, and target metadata with no targets. The viewer
reads an absent key and an empty one the same way. The skip also keeps the
committed writer fixture unchanged.

## What `meta` carries now

The bundle gains a `presentation` section: `scenario_replay_metadata(scenario)`,
lifted verbatim out of `_build_replay_metadata`. It is the scenario half of the
reference's `meta`: the terrain summary and the viewer's terrain mesh,
obstacles, land cover and its legend, CRS, bounds, scenario options, zones,
the ENU origin, and the motion, planner, target and role assignments. The Rust
writer copies it into `meta` and adds the run half itself:

* `platform` (the tracker's thresholds) and the requested and actual
  durations;
* `observation_generation`, stage 6's tallies summed over the replay's frames.

Two keys are deliberately left out. `performance` would make a replay's bytes
depend on the clock. `vio_tracked_nodes` is also left out: the reference lists
VIO backends even for runs that do not substitute VIO positions, and stage 5
is not assembled.

The lift was proven by a before/after run. Python replays of a
`scan_map_inspect` and a target-tracking scenario are identical in every
value and in `meta`'s key order. The only difference between runs is the key
order of the tracker-metric maps, which reach Python through gRPC from a Rust
`HashMap` and vary between any two runs.

## Bundle additions (all additive, all optional)

| Field | Why |
| --- | --- |
| `coverage_bounds` | `_coverage_bounds`: every mode keeps a coverage map and zone maps, but only `mission` named the grid |
| `targets[].behavior` | `target_motion_assignments`, for target metadata |
| `cooperative_search` | the team localization readout |
| `requested_duration_s` | `meta.requested_duration_s` |
| `presentation` | the scenario half of `meta` |

A bundle written before this slice still runs. Its target-tracking runs have
no stage 8, and so no mapping state or inspection events.

## New state, and the checkpoint

Two of the payloads needed state no stage kept before, and both are
checkpointed:

* **Stage 8, zones.** Each mission zone's coverage map, marked only while a
  drone is inside, and the set of drones inside. The event fraction is read
  after that drone's mark and before the next drone's, because the zone loop
  is the outer one. Restore checks the zone ids and each zone grid's bounds.
  The stage also keeps `last_estimates`, because the reference keeps
  reporting `_loc_estimates` after localization stops. `estimates` itself,
  which the mission machine reads, still empties. Finally it keeps the team
  fusion, because `_team_loc_current` persists too.
* **Stage 9, the coordinator.** `elect_coordinator` runs once, on the first
  scanning tick with a fleet: the highest battery fraction wins, and ties go
  to the first in fleet order.

Newly covered cells and zone events are per-tick output, not state.

## Findings

* **Deconfliction measured separation with `hypot`.** The reference uses
  `np.linalg.norm`. On a real resolution the reported predicted separation
  came out one ulp apart (12.663695267112088 against 12.66369526711209). That
  value sizes the lateral push, so the difference can move a platform. The
  kinematics comparisons had not caught it, because this push absorbed the
  ulp. `argusnet-planning`'s `norm2` is now `sqrt(x*x + y*y)`, the same fix
  P4.8e.3 made in the POI manager.
* **The reference serializes `meta` and frames differently.**
  `write_replay_document` passes frames through `to_jsonable`, which flattens
  2-D arrays, but dumps `meta` as it stands. So the terrain mesh's rows stay
  nested. The first version of this slice ran `presentation` through
  `to_jsonable`. The capture then compared it with a `to_jsonable` of the
  reference document, and the two agreed, both wrong. The viewer-reads-it test
  caught it: the viewer found no terrain. The capture and a Python test now
  compare against what the writer actually puts on disk.
* **The reference keeps stale localization estimates.** `scan_mission_state.
  localization_estimates` keeps the last localizing tick's estimates through
  inspection, egress and completion. The Rust mapping stage cleared them, and
  that was correct for the mission machine's read. The replay needed the
  sticky list, which is now a separate field.

## Verification

| Test | Shows |
| --- | --- |
| `replay_enrichment::a_truth_fed_run_reports_every_deterministic_payload_exactly` | all-follow case, 80 ticks: mapping state, inspection and deconfliction events and target metadata, bit for bit; the draw-free rejections are the same pairs; every pair both runs rejected at the same gate is identical in text and points, for all four gates the reference used (FOV 125, line of sight 82, detection draw 72, dropout draw 3) |
| `replay_enrichment::a_replayed_scan_mission_reports_its_state_exactly` | the default mission with localization replayed, 400 ticks: mapping state, zone events and the whole scan mission state, bit for bit (phase, coverage, POIs with positions, counts, phase start, newly scanned cells, timeout latch, coordinator, egress progress, team) |
| `replay_enrichment::the_replay_meta_carries_the_scenario_and_the_run` | presentation copied verbatim; platform and durations exact; attempted count exact; every generation total exact when no draw reaches (scan) |
| `replay_enrichment::numpy_mean_is_numpys_pairwise_mean` | bit-exact against numpy either side of the 8- and 128-element thresholds, where a left-to-right sum differs |
| `mapping::tests::zone_events_are_edges_and_an_exclusion_entry_is_a_violation`, `a_zone_event_reads_coverage_between_drones`, `cooperative_search_reports_a_team_estimate_and_keeps_it`, and the extended phase-gate test | the zone and team semantics the fixtures do not reach: violations, the inclusive radius, the between-drones fraction, the team's contributors, and the sticky estimates |
| `assembled_resume::a_checkpoint_for_a_different_scenario_is_refused_stage_by_stage` | two new refusals: different zones, and a zone map over a different grid |
| `cli::the_viewer_reads_everything_an_argusnet_rs_replay_carries` | the viewer's own `argusnet-replay` types parse an `argusnet-rs` replay and find the terrain mesh, zones, stale threshold and every panel's payload |
| `cli::a_checkpoint_resumed_from_the_command_line_continues_the_same_run` | unchanged, and now covers the new state too: resumed frames equal the originals including zone events, coordinator and scan state |
| `test_an_argusnet_rs_replay_passes_the_schema_with_every_payload` | strict `jsonschema` validation of enriched Rust replays; `meta` holds every key Python's does except the two declined |
| `test_the_bundles_presentation_is_the_replay_writers_meta` | the exported presentation is what `write_replay_document` writes, rows nested |
| `the_traced_cascade_names_each_gate` | `evaluate_pair_traced` names all seven gates, including the two that both report `dropout` |

### Mutation round

22 mutants. After one strengthening pass: **21 caught, and 1 was dead code,
now removed.** The first pass caught 15. One mutant was malformed and was
rewritten. Six survived, each because no fixture reached the behaviour:

| Survivor | What closed it |
| --- | --- |
| newly scanned cells not sorted | a unit test where the later drone covers the lower `i` |
| team fusion without its tightness filter | the team test now checks contributors on every tick, and requires a tick where one drone was fused and the other was not |
| the coordinator re-elected every scanning tick | a driver test that drains the coordinator's battery and steps again |
| priority 2 for index 1 only | `target_priority` extracted and pinned |
| the export ignoring the scenario's coverage resolution | a Python test at 25 m, in both modes |
| newly scanned cells not deduplicated | **equivalent: the `dedup` was dead code.** A cell is recorded only while its count is zero, and its first mark ends that. Removed, with the reason in a comment |

The caught mutants cover: the dropout and line-of-sight detail texts, a dropped
closest point, the detection gate's text, the zone radius made exclusive, a
violation reported as an entry, the zone fraction read before the mark, zone
membership not restored, non-sticky estimates, the coordinator's tie going to
the last drone, `hypot` in egress and in deconfliction, no Python rounding, a
sequential mean, the actual duration off by one tick, and a flattened
presentation.

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1060 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `ruff check src/argusnet scripts tests`, `ruff format --check` on touched files | clean |
| `pytest tests/` | 1022 passed, 9 skipped, 0 failed |
| CI | **not run**: nothing is pushed |

## Not in this slice

* **Order within a frame's observation and generation-rejection lists.** Stage
  6 visits sensors in id order, and the reference visits them in scenario node
  order. Consumers key by node and target, and the tests compare that way.
* **Safety events and VIO pose estimates**, until stages 4 and 5 are assembled.
* **`meta.performance`**, deliberately; the CLI prints its timing instead.
* **A Rust-built presentation.** The viewer's terrain mesh still comes from
  Python through the bundle. Building it in Rust belongs to P4.9b.

## Next

**P4.9b**: a Rust-native scenario builder. Parity is the identical bundle,
now including `presentation`.
