# PLANNING.md — Mapping, Localization, Inspection, and Coordination Planning

This document describes ArgusNet planning for the current product direction: map an area, localize
platforms within that map, inspect map-relative POIs, coordinate multiple drones, and return safely.

## Current Planner Modules

| Capability | Current implementation | Runtime status |
|------------|------------------------|----------------|
| 2D obstacle-aware route planning | `src/argusnet/planning/planner_base.py` (`PathPlanner2D`, `PlannerConfig`, `PlannerRoute`) | Used by simulation controllers and route helpers |
| Coverage/frontier utilities | `src/argusnet/planning/frontier.py` | `find_gap_cells()` is wired into `scan_map_inspect`; `select_frontier_cell()` is a library helper |
| Multi-drone coordination helpers | `src/argusnet/planning/coordination.py` | Coordinator election is wired; RF-delayed claims and formations are library helpers |
| Inspection POI assignment | `src/argusnet/planning/poi.py` | Wired into inspection assignment, dwell, handoff, team assignment, and rescoring |
| Deconfliction | `src/argusnet/planning/deconfliction.py` | Wired into current sim deconfliction events |
| Mission generation | `src/argusnet/planning/inspection.py` | Generates map/inspection mission structures and validation reports |

## Planning Objectives

ArgusNet planning optimizes for useful map and inspection work under safety constraints:

- **Coverage progress**: observe unvisited or under-observed map cells inside the mission boundary.
- **Localization confidence**: prefer actions that improve platform pose confidence before precise
  map-relative navigation.
- **Inspection value**: prioritize high-value POIs, overdue revisits, and POIs in poorly covered
  regions.
- **View quality**: choose approach paths and standoff positions that give useful sensor geometry.
- **Energy margin**: preserve return-home reserve before accepting new mapping or inspection work.
- **Comms continuity**: keep required peer/relay links healthy when coordination depends on them.
- **Terrain and obstacle clearance**: reject or reroute unsafe plans.

Ground truth can be used for simulation scoring after the fact. Planning should use mapping and
localization state, not hidden simulation truth.

## Planner-To-Trajectory Contract

A route becomes executable only after it has enough context for safety and replay:

```text
Mapping / inspection intent
  -> candidate POI, frontier, revisit, hold, or return-home task
  -> candidate route or viewpoint
  -> altitude and speed profile
  -> safety/deconfliction checks
  -> executable trajectory
  -> replay/evaluation event
```

Every installed trajectory should record:

- drone ID;
- task type (`map_frontier`, `localize`, `inspect_poi`, `revisit_poi`, `return_home`, `hold`);
- route or viewpoint source;
- planned timestamp;
- expected duration;
- safety validation result;
- reason for rejection or override, when applicable.

## Coverage And Frontier Planning

`FrontierPlanner` provides two separate capabilities:

- `find_gap_cells(cmap)` is currently wired into `scan_map_inspect` as the scan-to-localization
  gate. It detects enclosed holes in the coverage map and prevents early transition when coverage is
  misleading.
- `select_frontier_cell(cmap, drone_xy, claimed, drone_id)` is a library helper for future per-drone
  next-cell selection. It is not currently called by `src/argusnet/simulation/sim.py`.

`ClaimedCells` is instantiated by the simulation, but current routing does not yet use it for
frontier assignment. A future wiring pass should connect selected cells to route generation and
shared claim updates.

## Multi-Drone Coordination

Current coordination is intentionally lightweight:

- `CoordinationManager.elect_coordinator()` is called once during scanning to choose a coordinator
  by available battery.
- `CoordinationManager.update_claimed()` and `flush_messages()` model RF-delayed claim updates, but
  `sim.py` does not currently call them.
- `CoordinationManager.formation_offsets()` computes line/V offsets, but `sim.py` does not
  currently apply formation offsets to mapping routes.

Future coordination work should frame these helpers around map coverage distribution, localization
support, inspection workload sharing, and safe separation.

## Inspection POI Assignment

`POIAssignmentContext` is the current energy-aware assignment input:

```text
POIAssignmentContext
  drone_id: str
  drone_pos: np.ndarray
  battery_remaining_wh: float
  battery_capacity_wh: float
  cruise_speed_mps: float
  battery_reserve_fraction: float
  timestamp_s: float
```

`POIManager.assign_energy_aware(context)` scores pending POIs using effective priority and travel
energy. Drones that cannot reach a POI while preserving reserve are skipped.

Other current POI features:

- `trigger_handoff(from_drone_id, to_drone_id)` transfers an active POI while preserving dwell.
- `request_team_assign(drone_id, poi_id)` lets multiple drones contribute dwell to the same POI.
- `rescore_from_map(coverage_map)` raises priority for POIs in poorly covered cells.

## Deconfliction And Safety

Deconfliction is evaluated before a route/viewpoint should be considered safe:

- preserve minimum drone-to-drone separation;
- avoid mission exclusion zones and blocked regions;
- keep terrain clearance above mission limits;
- hold or reroute when a corridor or local area is temporarily occupied;
- record `DeconflictionEvent` when a resolution is applied.

The current safety posture is partly logging-oriented. Future runtime work should route all motion
through a blocking safety gate before execution.

## Roadmap

1. Wire `select_frontier_cell()` and `ClaimedCells` into actual coverage routing.
2. Connect RF-delayed shared claims to the mapping planner only after deterministic tests exist.
3. Replace old role-oriented planning language with task types: mapping, localization support,
   inspection, revisit, hold, and return-home.
4. Add route/viewpoint event logs that feed `PERFORMANCE_AND_BENCHMARKING.md` metrics.
5. Move toward a full closed-loop mission executive where planning proposes and safety validates
   every action.
