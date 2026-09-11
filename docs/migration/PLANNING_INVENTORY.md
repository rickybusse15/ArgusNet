# Planning inventory — P4.7b

P4.4 opened with an instruction rather than a task — *"inventory which helper
modules are actually wired before porting stubs"* — and that inventory more
than halved the job. This is the same exercise for `src/argusnet/planning`,
and it does the same thing: **the port surface is 39% of the package.**

It also corrects a premise this migration has been carrying since Phase 0.

Everything below was measured or derived by following imports, not by grepping
for names. Both tools are committed (`scripts/module_reachability.py`,
`scripts/module_symbols.py`) so the numbers are reproducible.

## Module reachability

Transitive import closure from the four `[project.scripts]` entry points
(`argusnet.cli.main`, `argusnet.simulation.sim`, `argusnet.evaluation.export`,
`argusnet.cli.scene`), versus the closure from `tests/`.

| module | lines | reachability |
|---|---:|---|
| `inspection.py` | 1470 | runtime |
| `coverage.py` | 615 | **test-only** |
| `planner_base.py` | 508 | runtime |
| `deconfliction.py` | 396 | runtime |
| `poi.py` | 293 | runtime |
| `battery.py` | 249 | runtime (ported in P4.7a) |
| `coordination.py` | 196 | runtime |
| `frontier.py` | 135 | runtime |
| `__init__.py`, `next_best_view.py`, `revisit.py` | 17 | orphan |

Runtime 3,247 · test-only 615 · orphan 17.

`coverage.py` is misnamed — it holds `CooperativePlanner` and
`PlanningObjectives`, not coverage — and it is reachable only from
`tests/test_cooperative_planner.py`. That misnaming is already recorded as a
Phase 0d finding; what is new is that nothing in the product imports it.

## Symbol reachability, and why the module table is not enough

**Nothing in the package is dead.** Every private symbol has a caller inside
its own module. The distinction that matters is runtime versus test-only, and
at module granularity it is invisible for the largest file.

### `inspection.py`: 1,470 lines, of which 144 reach production

| symbol | lines | referenced by |
|---|---:|---|
| `MissionSpec` | 69 | `mission/profiles.py` |
| `FlightCorridor` | 31 | `planning/deconfliction.py` |
| `MissionConstraints` | 25 | `mission/{__init__,execution,profiles}.py`, `simulation/sim.py` |
| `MissionTiming` | 19 | `mission/profiles.py` |
| `validate_mission` | 238 | `tests/test_mission_gen.py` only |
| `generate_mission` | 127 | `tests/test_mission_gen.py` only |
| `apply_difficulty_scaling` | 109 | `tests/test_mission_gen.py` only |
| four `*_template` functions | 247 | `tests/test_mission_gen.py` only |
| `ValidityReport`, `MissionObjective`, `GeneratedMission`, `ObjectiveCondition`, `_base_tags`, `_difficulty_band` | 144 | `tests/test_mission_gen.py` only |
| ten private `_build_*` helpers and `LaunchPoint` | 303 | internal to the generator above |

So production uses **four dataclasses**; the mission *generator* — roughly
1,326 lines — is exercised only by its own test file. Checked two ways beyond
the import graph: `mission/profiles.py` constructs a `MissionSpec` by hand at
`profiles.py:348` rather than calling `generate_mission`, and no CLI
subcommand exposes mission generation.

This is the same shape as P4.4's `state.py`: **test-only, not dead.** It has
real tests and may be a partial implementation of something wanted. Phase 6
owns whether it survives; what is established here is that it is not
load-bearing today and does not belong in the P4.7 port.

### The rest

| module | production surface |
|---|---|
| `planner_base.py` | `PathPlanner2D` (370), `PlannerRoute` (12), `PlannerConfig` (7), four private helpers (60). `_point_on_polygon_boundary` (7) is test-only. |
| `poi.py` | `POIManager` (255), `POIAssignmentContext` (10), `_xy` (9) — all runtime. |
| `deconfliction.py` | `DeconflictionLayer` (176), `DeconflictionConfig` (7), plus 163 private lines including `CorridorWindowAllocator`, instantiated at `deconfliction.py:243`. |
| `coordination.py` | all 165 lines runtime. |
| `frontier.py` | `FrontierPlanner` (95), `ClaimedCells` (14), `FrontierConfig` (9) — all runtime. |

**P4.7's port surface is ~1,496 production lines of the package's 3,879**,
excluding the battery model already delivered.

## The route cache omits the scene revision

P4.7 asks to *"cache routes against actual input revisions and quantized
request keys; never reuse a route across a changed exclusion zone blindly."*

`PathPlanner2D._route_cache_key` is:

```python
(quantized start_x, quantized start_y, quantized goal_x, quantized goal_y,
 round(clearance_m, 3))
```

Quantized request key, no input revision. It carries no obstacle or
exclusion-zone identity of any kind. And `PathPlanner2D.clear_cache()` has
**no caller anywhere** in `src/` or `tests/` — the only other `clear_cache`
hits in the repository are the world's unrelated `clear_caches`.

The counter the key needs already exists, and the pattern is already
established one module over. `Environment.add_obstacle()` swaps in a fresh
`ObstacleLayer` and calls `bump_scene_version()`, and `EnvironmentQuery.los`
keys its cache on `(scene_version, quantized geometry, profile)` —
`visibility.py:209`. The planner is the one cache in the world/planning stack
that does not consult it.

**Latent, not live.** `add_obstacle`, `set_terrain` and `bump_scene_version`
have no callers outside `environment.py`, so nothing can change the exclusion
set mid-run today. Note also that clearing the cache alone would not be
enough: the planner holds the `ObstacleLayer` instance passed at construction
and snapshots `_hard_primitives` from it, so after `add_obstacle` replaces the
layer the planner would still be routing around the old obstacle set. The port
should key on the revision rather than reproduce the omission.

## Correction: the "12 route plans per drone per tick" premise is wrong

The standing plan attributes a hot spot to `FollowPathController`: *"can fire
up to 12 raycasts **and** 12 route plans per drone per tick."* The structure
is there — a `for index in range(self.candidate_count)` loop with
`candidate_count: int = 12`, calling `plan_route` and `query.los` per
candidate. But the route call sits behind `if self.planner is not None:`, and

**`FollowPathController.planner` is `None` at every construction site.**

`sim.py` has exactly two `follow_path(...)` calls, at lines 3796 and 3832, and
both pass `planner=None`; `follow_path`'s own parameter defaults to `None`;
`sim.py` is the only file that constructs the controller at all. The
`planner=planner` at line 3829 belongs to a neighbouring `search_path(...)`
call, not to a `follow_path(...)` one. **The route half of that loop never
executes in the shipping simulation.** The raycast half is not guarded and
does run.

Measured, 40 ticks × 3 drones, seed 7, by counting calls through the real
`run_simulation`:

| configuration | `plan_route` | cache | visibility graph + A\* |
|---|---|---|---|
| `scan_map_inspect`, default | **0** | — | 0 |
| `scan_map_inspect`, `occlusion_aware_mapping` + frontier | 120 (**3.00/tick = 1/drone/tick**) | 110 hit / 10 miss — **91.7%** | — |
| `target_tracking`, `occlusion_aware_mapping` | 14 total (0.35/tick), all construction-time | 0 hit / 14 miss | **1 of 14** |

Three further readings worth keeping:

- **Route planning is gated on `--occlusion-aware-mapping`**, the same gate
  Phase 3 found for `EnvironmentQuery.los`. In the default configuration the
  planner is not called at all after construction.
- **The expensive path is rare.** Only 1 of the 14 `target_tracking` routes
  needed a visibility graph and an A\* search; the other 13 resolved on the
  straight-line check.
- **LOS is the real per-tick cost in this area**, at **44.5 calls per drone
  per tick** — across sensing, mapping and the candidate loop, not the planner.

So P4.7's exit criterion — *"the mission loop no longer recomputes multiple
identical routes per platform per tick"* — describes a configuration that does
not ship. What is true is narrower and worth stating as the actual target: the
redirect planner is called once per drone per tick under occlusion-aware
mapping, its existing cache already absorbs 92% of that, and the remaining
defect is the missing revision guard above rather than redundant computation.

Porting this as-measured, rather than as-assumed, avoids building a caching
layer against a cost that is not there.

## Consequences for the rest of P4.7

- Port `planner_base`, `poi`, `deconfliction`, `frontier` and `coordination`
  (~1,496 lines). Take `MissionSpec`, `MissionConstraints`, `MissionTiming`
  and `FlightCorridor` from `inspection.py` as types; leave the generator.
- Key the route cache on the scene revision, since the counter exists and the
  reference already uses it elsewhere. This is a behaviour change under
  ADR-006, not a parity item — record it rather than inherit the omission.
- Do not build the deduplication the exit criterion implies until a
  configuration is shown to need it. The measurement above is the baseline any
  such claim has to beat.
