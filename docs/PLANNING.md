# PLANNING.md — Drone Role Model and Planner-to-Trajectory Contract

This document defines the cooperative mission planning model (Sections 9 and 10 of the
architecture update plan): drone roles, planning objectives, the contract that turns a
planned route into an executable trajectory, replanning triggers, and deconfliction rules.

References throughout point to the existing implementations in:
- `src/smart_tracker/planning.py` (`PathPlanner2D`, `PlannerConfig`, `PlannerRoute`)
- `src/smart_tracker/sim.py` (`ScenarioDefinition.drone_roles`, `drone_planner_modes`,
  `drone_target_assignments`, `adaptive_drone_controllers`)
- `src/smart_tracker/config.py` (`DynamicsConfig`, `SensorConfig`)
- `src/smart_tracker/behaviors.py` (`FlightEnvelope`, `TransitBehavior`, `LoiterBehavior`)

---

## 1. Drone Role Model

Every drone (mobile `SimNode`) is assigned exactly one role at scenario construction time.
The role is stored in `ScenarioDefinition.drone_roles: Mapping[str, str]` where the key is
`node_id`.

### Role Definitions

```
Role: "primary_observer"
  Description:
    Maintains close sensor coverage of the highest-priority active target.
    Orbit radius: DynamicsConfig.interceptor_follow_radius_m (55 m default).
    Altitude offset: DynamicsConfig.interceptor_follow_altitude_offset_m (35 m default).
  Planner mode: "follow"
  Replanning trigger: track handoff, track loss, or stale_steps >= 2
  Min count per mission: 1

Role: "secondary_baseline"
  Description:
    Orbits the same target at a wider standoff to maximise triangulation baseline.
    Orbit radius: DynamicsConfig.tracker_standoff_radius_m (120 m default).
    Altitude offset: DynamicsConfig.tracker_altitude_offset_m (70 m default).
  Planner mode: "follow" (tracker sub-variant)
  Replanning trigger: same as primary_observer
  Min count per mission: 0 (recommended >= 1 for adequate localisation)

Role: "corridor_watcher"
  Description:
    Patrols a FlightCorridor (from MISSION_MODEL.md) to detect and hand off targets
    entering or exiting the sensor coverage area.
    Speed: DynamicsConfig.drone_search_speed_base_mps (28 m/s default, scaled by platform).
  Planner mode: "search" (lawnmower or waypoint patrol along corridor centerline)
  Replanning trigger: corridor assignment change or new target entry detected
  Min count per mission: 0

Role: "relay"
  Description:
    Stationary or slow-loiter platform that extends comms range between drones beyond
    MissionConstraints.comms_range_m from any ground station.
    Altitude: chosen to maximise LOS to both ground station and furthest active drone.
  Planner mode: "search" (loiter at computed relay point)
  Replanning trigger: comm link quality drops below threshold or relay point becomes occluded
  Min count per mission: 0

Role: "reserve"
  Description:
    Grounded drone awaiting launch via LaunchEvent. Activated when an active drone
    exhausts energy reserve or is lost.
    Launch trigger: energy_reserve_fraction < 0.10 on any active drone with same target assignment.
  Planner mode: assigned on activation (inherits role of the drone it replaces)
  Replanning trigger: on launch completion
  Min count per mission: 0
```

Role assignments are validated at `GeneratedMission` construction time. At least one
`primary_observer` must be present. If `target_count > 1`, at least two drones with
distinct `drone_target_assignments` values must exist.

---

## 2. Planning Objectives

The planner optimises a weighted sum of the following objectives. Weights are configurable
per mission type; defaults shown in brackets.

```
PlanningObjective
  track_continuity        [1.0]  # Maximise fraction of [start_s, end_s] each required target
                                 # has >= 1 active bearing observation per step.
                                 # Drives drone positions to maintain sensor coverage.

  localisation_quality    [0.8]  # Minimise trace(P_position) for each active track.
                                 # Improves by widening inter-drone baseline.
                                 # Measured as covariance_trace from TrackState.covariance.

  geometric_diversity     [0.5]  # Maximise angular spread of bearing vectors to each target.
                                 # Minimum acceptable spread: pi/4 rad (45 degrees).
                                 # Computed as min pairwise angle across all observing nodes.

  persistence             [0.6]  # Penalise unplanned breaks in coverage (stale_steps > 0).
                                 # Heavier weight for required objectives.

  resilience              [0.4]  # Maintain fallback coverage when any single drone fails.
                                 # Requires at least one other drone able to observe each
                                 # target independently.

  terrain_clearance       [1.0]  # Hard constraint: drone AGL >= MissionConstraints.terrain_clearance_m.
                                 # Violation cost is infinite (infeasible path rejection).

  energy_reserve          [0.3]  # Penalise routes whose estimated flight time brings
                                 # projected energy below 0.15 (15 %) at mission end.

  comms_connectivity      [0.2]  # Penalise positions where any drone exceeds
                                 # MissionConstraints.comms_range_m from its nearest peer
                                 # or ground station.
```

The `terrain_clearance` objective is always weight 1.0 regardless of mission type and
cannot be downweighted.

---

## 3. Planner-to-Trajectory Contract

This section defines how a `PlannerRoute` from `PathPlanner2D` is converted into an
executable trajectory function (`TrajectoryFn: (float) -> (np.ndarray, np.ndarray)`).

### 3.1 Schema: PlannedTrajectory

```
PlannedTrajectory
  drone_id:             str
  route:                PlannerRoute          # Output of PathPlanner2D.plan_route() or
                                              # PathPlanner2D.route_waypoints()
  altitude_profile:     AltitudeProfile
  speed_mps:            float                 # Nominal cruise speed along the route
  role:                 str                   # Role at planning time (for audit)
  planned_at_s:         float                 # Simulation time when plan was computed
  valid_until_s:        float                 # Staleness deadline; see Section 4
  generation:           int                   # Monotonically increasing plan version counter
  override_reason:      str | None            # Non-None if this plan resulted from a safety override
```

```
AltitudeProfile
  mode:                 str                   # "fixed_agl" | "terrain_following" | "fixed_msl"
  base_agl_m:           float                 # Nominal AGL; maps to DynamicsConfig.drone_base_agl_m
  min_agl_m:            float                 # Floor; must be >= terrain_clearance_m
  max_agl_m:            float                 # Ceiling from FlightEnvelope.max_altitude_agl_m
  terrain_following_smoothing_s: float        # Low-pass time constant from DynamicsConfig
                                              # (default 1.5 s)
```

### 3.2 Conversion Steps

Given a `PlannerRoute` (2D XY waypoints) and an `AltitudeProfile`, the conversion pipeline
produces a `TrajectoryFn`:

1. **Waypoint lift**: Each 2D waypoint `(x, y)` in `PlannerRoute.points_xy_m` is elevated
   to 3D using `TerrainModel.height_at(x, y) + base_agl_m`. When `mode="terrain_following"`,
   the AGL is recalculated at each simulation step using the smoothing filter from
   `DynamicsConfig.terrain_following_smoothing_s`.

2. **Speed enforcement**: Assign speed `speed_mps` along each segment. If any segment
   requires a turn angle > 90 degrees, reduce local speed to satisfy `FlightEnvelope.min_turn_radius_m`
   (derived from `max_bank_angle_deg`).

3. **Trajectory function**: Construct a `TransitBehavior` (behaviors.py) from the lifted
   3D waypoints with the segment speeds. The returned callable is the `TrajectoryFn` stored
   on the `SimNode.trajectory`.

4. **End-of-route behaviour**: On reaching the final waypoint, the drone transitions to
   `LoiterBehavior` at the terminal position unless a new `PlannedTrajectory` has been
   issued before `valid_until_s`.

5. **Safety gate**: Before installing the new trajectory, assert:
   - `route` is not `None` (PathPlanner2D returned a valid path).
   - All lifted 3D points satisfy `z >= TerrainModel.height_at(x, y) + min_agl_m`.
   - No point falls inside any obstacle from `ObstacleLayer.primitives`.
   If any assertion fails, the plan is rejected (increment `safety_override_count` in
   evaluation metrics) and the drone retains its previous trajectory.

6. **Smoothing**: The converted trajectory inherits path smoothing already applied by
   `PathPlanner2D._smooth_path()`. Additional velocity smoothing is provided by the
   cubic Hermite interpolation inside `TransitBehavior`.

### 3.3 Immutable Contract Properties

The following invariants must hold for every `PlannedTrajectory` installed on a drone:

- `planned_at_s <= valid_until_s` (plan is not born stale).
- `speed_mps` is within `[FlightEnvelope.min_speed_mps, FlightEnvelope.max_speed_mps]`.
- `base_agl_m >= min_agl_m`.
- `route.length_m > 0` and `route.vertex_count >= 2`.
- The drone's current position at `planned_at_s` is within `PlannerConfig.snap_m` of
  `route.points_xy_m[0]` (plans do not teleport the drone).

---

## 4. Timing: Replanning Triggers and Staleness Thresholds

### 4.1 Staleness Threshold

A `PlannedTrajectory` becomes stale when `simulation_time >= valid_until_s`. The default
`valid_until_s` is computed as:

```
valid_until_s = planned_at_s + max(30.0, route.length_m / speed_mps * 0.5)
```

That is, a plan is valid for at least 30 seconds, or half the estimated flight time,
whichever is larger. This prevents excessive replanning on short routes while ensuring
long routes are reconsidered as conditions change.

### 4.2 Replanning Triggers (priority order)

The following events trigger immediate replanning regardless of `valid_until_s`:

1. **Track loss** (`stale_steps >= DynamicsConfig.default_max_stale_steps` for the drone's
   assigned target). Drones with role `primary_observer` or `secondary_baseline` transition
   to a search sub-route centred on the last known position.

2. **Track handoff** — a different drone has a better geometry score (lower covariance trace)
   for the current target. The current drone is reassigned or transitioned to `secondary_baseline`.

3. **Obstacle ingress warning** — the look-ahead position `t + follow_lead_s` of the
   current `TrajectoryFn` would violate terrain clearance or enter an obstacle. Triggers
   emergency reroute via `PathPlanner2D.plan_route()`.

4. **New exclusion zone** — a `MissionZone` with `zone_type="exclusion"` is added at
   runtime. All drones whose current routes intersect the zone are replanned.

5. **Role reassignment** — the drone is given a new role by the cooperative planner
   (e.g., a `reserve` drone is activated to replace a failed `primary_observer`).

6. **Staleness expiry** — `valid_until_s` reached with no other trigger.

### 4.3 Replan Cooldown

To prevent oscillation, a drone may not be replanned more than once per 5 simulation
seconds, except for triggers 3 (obstacle ingress) and 5 (role reassignment), which
override the cooldown.

---

## 5. Deconfliction Rules

Deconfliction is evaluated in 2D XY space at each simulation step. The rules apply to
all active (non-`reserve`) drones simultaneously.

### 5.1 Separation Minimum

```
min_separation_m = 2 × PlannerConfig.drone_clearance_m   # default: 2 × 8.0 = 16 m
```

If any two drones are predicted to be within `min_separation_m` within the next
`follow_lead_s` seconds:
- The lower-priority drone (by role priority order: `primary_observer` > `secondary_baseline`
  > `corridor_watcher` > `relay` > `reserve`) yields.
- Yielding drone computes an alternate next waypoint offset 90 degrees from its current
  heading at the minimum separation distance and replans.

### 5.2 Corridor Conflict

If two drones are assigned the same `FlightCorridor` traveling in opposing directions
(`direction="bidirectional"`), they are allocated alternating time windows. Each window is
`corridor.length_m / speed_mps` seconds long. The planner inserts a loiter hold at the
corridor entry point for the lower-priority drone until the window clears.

### 5.3 Vertical Separation

When two drones must pass within `min_separation_m` in XY (unavoidable given terrain and
obstacles), the lower-priority drone adjusts altitude to maintain a vertical separation of
at least 20 m. This is recorded as a `comms_dropout` entry if the altitude change breaks
LOS to its ground station.

### 5.4 Exclusion Zone Enforcement

A drone inside or within `PlannerConfig.drone_clearance_m` of an exclusion zone boundary
is immediately rerouted. `PathPlanner2D` treats exclusion zones as hard obstacles by
adding them to `ObstacleLayer` as `CylinderObstacle` primitives with radius
`MissionZone.radius_m`.
