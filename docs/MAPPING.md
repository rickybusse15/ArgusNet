# MAPPING.md — Belief-World Mapping and Guidance Contract

This document defines the mapping step for ArgusNet: how the system should build a responsible, accurate, and useful understanding of an environment when no complete map or ground truth exists.

ArgusNet should not assume that a prior map is available in physical deployments. In simulation, ground truth may exist for scoring and debugging, but real drones must construct a working world model from observations. That reconstructed model becomes the basis for guidance, trajectory planning, safety validation, and coverage decisions.

---

## 1. Purpose

The mapping subsystem turns sensor observations into a continuously updated belief about the world.

It answers:

- What parts of the environment have been observed?
- What terrain has been reconstructed?
- What areas are still unknown?
- What is safe enough to fly through?
- Where should the drone observe next?
- When should mapping stop?

The output of mapping is not just a visual mesh. It is a planning and safety input.

> ArgusNet plans from the current belief of the world, not from simulation ground truth.

---

## 2. Why this matters to ArgusNet

ArgusNet is intended to support autonomous aerial mapping, inspection, localization, and spatial memory. Those goals require the system to build a usable world model over time rather than relying on a complete environment description at mission start.

This is especially important for physical deployments because:

- ground truth terrain is not available;
- obstacles may be unmapped, temporary, or changed;
- launch areas may only have a rough geofence or mission boundary;
- guidance must be conservative where uncertainty is high;
- the viewer should show the same reconstructed world that the planner is using;
- evaluation in simulation must verify that the planner did not cheat by using ground truth.

Mapping is therefore the bridge between sensing and planning. It converts raw observations into a queryable belief world that can support responsible guidance.

---

## 3. Core distinction: truth, prior, belief, and visual world

ArgusNet should maintain separate world representations with different authority levels.

| World representation | Meaning | Used by planner? | Used in physical deployment? |
|----------------------|---------|------------------|-------------------------------|
| Ground Truth World | Complete simulated environment used for scoring and generating synthetic observations | No | No |
| Prior World | Optional starting information such as launch point, rough DEM, known geofence, imported map, or site boundary | Yes, as initial belief only | Yes |
| Belief World | Reconstructed terrain, obstacles, coverage, and uncertainty from live observations | Yes, authoritative for planning | Yes |
| Visual World | Viewer mesh and UI representation of the belief or replay state | No direct authority | Yes, for operator understanding |

Ground truth must remain isolated from mission planning during simulation. It may be used to generate observations and compute evaluation metrics, but it must not be used by the planner as if it were known.

---

## 4. Initial mission boundary

A physical mission should begin with a bounded objective. The minimum initial inputs are:

- launch position;
- geofence polygon or radius around launch;
- maximum mapping radius or selected mission area;
- return-home location;
- maximum altitude;
- minimum terrain clearance;
- battery reserve requirement;
- sensor configuration;
- coverage or inspection objective.

Example mission statement:

```text
Map all reachable terrain within 500 m of launch,
stay inside the selected geofence,
remain below the legal/mission altitude limit,
maintain minimum terrain clearance,
and return with the required battery reserve.
```

The geofence is not just a visual boundary. It is a hard planning and safety constraint. Unknown space inside the geofence may be explored cautiously. Space outside the geofence is forbidden unless explicitly authorized by a new mission command.

---

## 5. Belief World Model

The mapping subsystem should output a `BeliefWorldModel`: a queryable representation of the current reconstructed environment.

At minimum, each map cell or local region should track:

- height estimate;
- height uncertainty;
- observed/unobserved state;
- obstacle probability;
- terrain confidence;
- land-cover or semantic class, if available;
- last observed timestamp;
- source observations or keyframes;
- coverage count or coverage quality;
- whether the cell is inside the mission geofence.

A cell with no observations is not the same as a safe cell. Unknown areas require explicit risk handling.

---

## 6. Planning behavior in known and unknown space

The planner should use the belief world with conservative rules.

| Region type | Planner behavior |
|-------------|------------------|
| Known safe | Route normally if clearance and constraints are satisfied |
| Known obstacle | Avoid as a hard blocker |
| Known terrain with high uncertainty | Route cautiously, increase clearance, or observe before entering |
| Unknown inside geofence | Explore through frontier planning with conservative speed and altitude |
| Unknown outside geofence | Treat as forbidden |
| Low-confidence reconstruction | Prefer additional observation before committing to aggressive paths |

This prevents the system from treating absence of data as evidence of safety.

---

## 7. Mapping loop

The physical mapping loop should operate as a closed cycle:

```text
Initial geofence / launch constraints
        ↓
Drone sensor observations
        ↓
Mapping update
        ↓
BeliefWorldModel update
        ↓
Coverage and uncertainty evaluation
        ↓
Frontier / next-best-view selection
        ↓
Trajectory planning against belief world
        ↓
Safety validation
        ↓
Drone motion and new observations
        ↓
Repeat
```

The loop ends when one or more stop conditions are met:

- required coverage reached;
- battery reserve threshold reached;
- no safe frontier remains;
- geofence boundary prevents further expansion;
- operator stops the mission;
- safety monitor triggers return-to-home or abort.

---

## 8. Frontier planning and next-best-view

When the map is incomplete, ArgusNet should choose actions that improve the belief world.

A frontier is a boundary between known and unknown space. Frontier planning selects safe, valuable frontier cells inside the geofence.

A frontier or next-best-view score should consider:

- expected new coverage;
- uncertainty reduction;
- distance and energy cost;
- terrain clearance risk;
- obstacle probability;
- communications connectivity;
- return-home feasibility;
- sensor viewing geometry;
- inspection priority, if a mission target exists.

This makes mapping active rather than passive. The drone does not only fly a fixed path; it uses the current reconstructed world to decide what information is most valuable next.

---

## 9. Viewer reconstruction role

The viewer should display the same belief-world layers that planning uses. It should not be treated as only a replay renderer.

Important viewer layers include:

- reconstructed terrain mesh;
- terrain uncertainty heatmap;
- observed vs unobserved regions;
- current geofence;
- planned path;
- return-home path;
- frontier cells;
- obstacle probability;
- unsafe or blocked regions;
- drone pose and sensor footprint;
- mapping progress metrics.

The visual mesh may be lower resolution than the analytic belief used by planning, but it should represent the same underlying state. Visual terrain should not become the authority for physics or safety; it is an operator-facing view of the belief model.

---

## 10. Required query interface

Planning and safety should query the belief world through an explicit interface rather than reading raw map arrays directly.

Suggested interface:

```text
WorldBeliefQuery
  height_estimate_at(x, y) -> value
  height_uncertainty_at(x, y) -> value
  obstacle_probability_at(x, y) -> value
  coverage_at(x, y) -> value
  confidence_at(x, y) -> value
  is_observed(x, y) -> bool
  is_inside_geofence(x, y) -> bool
  is_known_safe(x, y) -> bool
  frontier_cells() -> list
  safe_corridor_between(a, b) -> route candidate
```

The existing analytic terrain interface remains useful for simulation and prior terrain, but physical planning should depend on `WorldBeliefQuery` or an equivalent runtime belief interface.

---

## 11. Integration with the existing architecture

Mapping should sit between sensing/localization and planning.

```text
Sensing / Perception
        ↓
Localization
        ↓
Mapping
        ↓
Belief World Model
        ↓
Planning
        ↓
Trajectory
        ↓
Safety Monitor
        ↓
Execution
```

Simulation may still maintain a ground truth world, but that truth should be connected only to:

- synthetic sensor generation;
- evaluation metrics;
- replay comparison;
- debugging tools.

The planner should receive the same kind of belief-state input that it would receive in a physical deployment.

---

## 12. State ownership

Recommended ownership:

| State | Owner | Notes |
|------|-------|-------|
| Ground truth terrain | Simulation only | Not available to planning |
| Prior mission boundary/geofence | Mission subsystem | Hard constraint |
| Belief terrain | Mapping subsystem | Authoritative planning input |
| Belief obstacles | Mapping subsystem | Probabilistic until confirmed |
| Coverage map | Mapping subsystem | Used by planner and evaluation |
| Viewer mesh | Viewer/UI subsystem | Derived display artifact |
| Safety constraints | Safety subsystem | Final gate before execution |
| Evaluation truth comparison | Evaluation subsystem | Simulation-only scoring |

---

## 13. Safety rules for physical mapping

Physical mapping requires conservative behavior around uncertainty.

Minimum rules:

1. The drone must remain inside the geofence.
2. The drone must preserve a valid return-home path or reserve policy.
3. Unknown space must not be treated as safe by default.
4. Low-confidence terrain requires increased altitude, slower motion, or additional observation.
5. The safety monitor may reject any planned trajectory even if the planner selected it.
6. Ground truth must never be used by planning during simulation tests intended to represent real deployment.
7. Mapping should stop or return home if safe exploration cannot continue.

---

## 14. Implementation priorities

### Phase 1 — Belief map foundation

- Add a `BeliefWorldModel` data structure.
- Track observed/unobserved cells.
- Track coverage and uncertainty.
- Store geofence membership.
- Expose a `WorldBeliefQuery` interface.

### Phase 2 — Viewer integration

- Render reconstructed terrain from the belief model.
- Add observed/unobserved overlays.
- Add uncertainty heatmap.
- Show geofence, frontiers, planned route, and return route.

### Phase 3 — Planning integration

- Make the planner consume `WorldBeliefQuery`.
- Add frontier selection inside geofence.
- Add risk-aware routing through uncertain terrain.
- Require safety validation before execution.

### Phase 4 — Simulation truth isolation

- Ensure simulation planners cannot read ground truth during physical-mode tests.
- Add evaluation checks that compare belief reconstruction against truth after the mission.
- Report map completeness, terrain error, obstacle detection performance, and unsafe-route attempts.

### Phase 5 — Physical runtime path

- Add live sensor ingestion into mapping.
- Add incremental terrain reconstruction.
- Add online replanning from belief updates.
- Add operator-supervised mission controls.

---

## 15. Success criteria

This mapping step is successful when ArgusNet can:

- start with only a launch point and geofence;
- build a reconstructed terrain and coverage map from observations;
- distinguish known safe, unknown, and unsafe regions;
- plan new trajectories from the belief world;
- display the same reconstructed world to the operator;
- stop at geofence, battery, or safety limits;
- evaluate reconstruction quality in simulation without allowing the planner to use ground truth.

At that point, ArgusNet becomes a true world-modeling system rather than only a tracker or replay simulator.
