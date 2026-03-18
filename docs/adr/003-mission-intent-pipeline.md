# ADR-003: Mission Intent Must Pass Through Feasibility Pipeline

**Status:** proposed
**Date:** 2026-03-15
**Author:** architecture-update
**Supersedes:** none

## Context

The architecture update (Section 2) defines the central design rule:

> Do not execute mission intent directly.

Currently, `sim.py` generates waypoints and drone positions directly. Drones follow waypoint lists without explicit feasibility validation, constraint checking, or trajectory smoothing. This works for simulation but prevents realistic multi-drone constrained operations.

## Decision

All mission actions must pass through:

```
mission intent → candidate route → feasibility validation → executable trajectory → safety monitor → execution
```

No drone may move without passing through this pipeline. The pipeline enforces:
- Physical constraint validation (speed, turn rate, climb rate, terrain clearance)
- Drone-to-drone deconfliction
- Energy budget check
- Communications coverage check
- Safety monitor approval

## Consequences

### Positive
- Prevents unrealistic behavior and hidden constraint violations
- Enables meaningful evaluation of planner quality
- Safety monitor can halt unsafe trajectories

### Negative
- All existing waypoint-based motion must be refactored
- Adds latency to the planning loop
- More complex simulation initialization

### Migration
1. Define `CandidateRoute` and `ExecutableTrajectory` types
2. Implement `FeasibilityValidator` with constraint checks
3. Implement `SafetyMonitor` as execution gate
4. Refactor `sim.py` drone motion to use pipeline
5. Update tests to verify constraint enforcement

## Affected Modules

| Module | Change type |
|--------|------------|
| `src/smart_tracker/sim.py` | modified (drone motion refactor) |
| `src/smart_tracker/planning.py` | modified (output CandidateRoute) |
| `rust/trajectory-engine/` | new crate |
| `rust/safety-engine/` | new crate |
| `src/smart_tracker/models.py` | modified (new types) |

## Tests Required

- Infeasible route rejection (exceeds speed, turn rate, etc.)
- Safety monitor halt on constraint violation
- End-to-end pipeline: intent → route → trajectory → execution
- Regression: existing simulation still produces valid replays

## References

- Architecture update Section 2: Core Architectural Principle
- Architecture update Section 10: Physical Constraints and Trajectory Execution
- Architecture update Section 10.3: Architectural Rule
