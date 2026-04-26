"""Cooperative mission planning model for the Smart Trajectory Tracker.

Implements the drone role model, planning objectives, planner-to-trajectory
contract, replanning triggers, and deconfliction rules described in
``docs/PLANNING.md`` (Sections 9 and 10 of the architecture update plan).

Typical usage::

    from argusnet.planning.coverage import (
        CooperativePlanner,
        PlanningObjectives,
        ROLE_PRIMARY_OBSERVER,
    )

    objectives = PlanningObjectives()
    planner = CooperativePlanner(objectives)
    roles = planner.assign_roles(
        drone_ids=["d0", "d1", "d2"],
        target_ids=["t0"],
        drone_roles={"d0": ROLE_PRIMARY_OBSERVER, "d1": ROLE_SECONDARY_BASELINE, "d2": ROLE_RESERVE},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Role constants
# ---------------------------------------------------------------------------

ROLE_PRIMARY_OBSERVER = "primary_observer"
ROLE_SECONDARY_BASELINE = "secondary_baseline"
ROLE_CORRIDOR_WATCHER = "corridor_watcher"
ROLE_RELAY = "relay"
ROLE_RESERVE = "reserve"

ROLE_PRIORITY: Dict[str, int] = {
    ROLE_PRIMARY_OBSERVER: 5,
    ROLE_SECONDARY_BASELINE: 4,
    ROLE_CORRIDOR_WATCHER: 3,
    ROLE_RELAY: 2,
    ROLE_RESERVE: 1,
}

# All known roles (used for validation)
_ALL_ROLES = frozenset(ROLE_PRIORITY)

# Replanning triggers that bypass the cooldown window
_COOLDOWN_BYPASS_TRIGGERS = frozenset({"obstacle_ingress", "role_change"})

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AltitudeProfile:
    """AGL / MSL altitude constraints for a planned trajectory segment."""

    mode: str
    """One of ``"fixed_agl"``, ``"terrain_following"``, or ``"fixed_msl"``."""

    base_agl_m: float
    """Nominal above-ground-level cruise altitude, in metres."""

    min_agl_m: float
    """AGL floor; must be >= terrain_clearance_m (hard constraint)."""

    max_agl_m: float
    """AGL ceiling derived from ``FlightEnvelope.max_altitude_agl_m``."""

    terrain_following_smoothing_s: float = 1.5
    """Low-pass filter time constant for terrain-following mode, in seconds.

    Maps to ``DynamicsConfig.terrain_following_smoothing_s`` (default 1.5 s).
    """


@dataclass(frozen=True)
class PlannedTrajectory:
    """Planner-to-trajectory contract for a single drone.

    Immutable record of the route, altitude profile, speed, and metadata
    associated with one planning cycle output.  The ``generation`` counter
    increments monotonically across the lifetime of the mission.
    """

    drone_id: str
    route: object  # PlannerRoute from planning.py, or None when no route computed
    altitude_profile: AltitudeProfile
    speed_mps: float
    role: str
    planned_at_s: float
    valid_until_s: float
    generation: int
    override_reason: Optional[str] = None


@dataclass(frozen=True)
class PlanningObjectives:
    """Weighted planning objectives for cooperative mission optimisation.

    All weights are in [0.0, 1.0].  ``terrain_clearance`` is always 1.0
    and cannot be downweighted — it is a hard feasibility constraint.
    """

    track_continuity: float = 1.0
    """Maximise fraction of mission window where each target has >= 1 bearing."""

    localisation_quality: float = 0.8
    """Minimise covariance-trace for each active track (wider baseline is better)."""

    geometric_diversity: float = 0.5
    """Maximise angular spread of bearing vectors; minimum acceptable: pi/4 rad."""

    persistence: float = 0.6
    """Penalise unplanned coverage gaps (stale_steps > 0)."""

    resilience: float = 0.4
    """Maintain fallback coverage when any single drone fails."""

    terrain_clearance: float = 1.0
    """Hard constraint: drone AGL >= terrain_clearance_m at all times."""

    energy_reserve: float = 0.3
    """Penalise routes whose estimated flight time exhausts energy below 15 %."""

    comms_connectivity: float = 0.2
    """Penalise positions where any drone exceeds comms_range_m from its nearest peer."""

    def __post_init__(self) -> None:
        # terrain_clearance is always 1.0; silently enforce regardless of
        # what the caller passed so the invariant documented in PLANNING.md holds.
        object.__setattr__(self, "terrain_clearance", 1.0)


@dataclass(frozen=True)
class PlannerEvent:
    """Audit log entry for a single planner action."""

    timestamp_s: float
    drone_id: str
    event_type: str
    """One of ``"plan_issued"``, ``"plan_rejected"``, ``"safety_override"``,
    ``"replan_trigger"``, or ``"role_change"``."""

    trigger: str
    """Human-readable trigger reason (e.g. ``"track_loss"``, ``"staleness_expiry"``)."""

    generation: int
    route_length_m: Optional[float] = None
    override_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# CooperativePlanner
# ---------------------------------------------------------------------------


class CooperativePlanner:
    """Cooperative mission planner: role assignment, trajectory generation,
    replanning decisions, and deconfliction.

    This class does not mutate any ``SimNode`` or ``ScenarioDefinition`` directly;
    it produces ``PlannedTrajectory`` objects and appends ``PlannerEvent``
    entries for downstream consumers to act on.

    Parameters
    ----------
    objectives:
        Weighted optimisation objectives for this mission.
    replan_cooldown_s:
        Minimum simulation seconds between replanning the same drone, unless
        an obstacle-ingress or role-change trigger overrides the cooldown.
    """

    def __init__(
        self,
        objectives: PlanningObjectives,
        replan_cooldown_s: float = 5.0,
    ) -> None:
        self._objectives = objectives
        self._replan_cooldown_s = float(replan_cooldown_s)
        self._plans: Dict[str, PlannedTrajectory] = {}
        self._events: List[PlannerEvent] = []
        self._generation_counter: int = 0
        self._last_replan_time: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_roles(
        self,
        drone_ids: List[str],
        target_ids: List[str],
        drone_roles: Dict[str, str],
    ) -> Dict[str, str]:
        """Validate role assignments and ensure at least one primary_observer.

        Returns the (possibly corrected) role mapping.  If no
        ``primary_observer`` is present in *drone_roles* the first drone
        in *drone_ids* is elevated to that role and a ``role_change`` event
        is recorded with trigger ``"forced_primary_assignment"``.

        Parameters
        ----------
        drone_ids:
            All drone node IDs for this mission.
        target_ids:
            All target IDs (used for multi-target validation).
        drone_roles:
            Proposed role mapping ``{drone_id: role_str}``.  Roles not in
            ``_ALL_ROLES`` are replaced with ``ROLE_RESERVE``.
        """
        result: Dict[str, str] = {}

        # Normalise and validate each entry
        for drone_id in drone_ids:
            raw_role = drone_roles.get(drone_id, ROLE_RESERVE)
            if raw_role not in _ALL_ROLES:
                raw_role = ROLE_RESERVE
            result[drone_id] = raw_role

        # Ensure at least one primary_observer
        has_primary = any(role == ROLE_PRIMARY_OBSERVER for role in result.values())
        if not has_primary and drone_ids:
            elected = drone_ids[0]
            result[elected] = ROLE_PRIMARY_OBSERVER
            self._record_event(
                PlannerEvent(
                    timestamp_s=0.0,
                    drone_id=elected,
                    event_type="role_change",
                    trigger="forced_primary_assignment",
                    generation=self._generation_counter,
                )
            )

        return result

    def plan_trajectory(
        self,
        drone_id: str,
        role: str,
        target_position: Optional[np.ndarray],
        drone_position: np.ndarray,
        planner: object,  # PathPlanner2D
        terrain_model: object,  # TerrainModel
        dynamics: object,  # DynamicsConfig
        timestamp_s: float,
    ) -> PlannedTrajectory:
        """Generate a ``PlannedTrajectory`` for *drone_id* based on its role.

        The planner computes a 2-D route using the supplied ``PathPlanner2D``
        instance and wraps it with an ``AltitudeProfile`` derived from
        *dynamics*.  The result is stored in ``self._plans`` and a
        ``"plan_issued"`` event is appended.

        If route planning fails (``PathPlanner2D.plan_route`` returns ``None``)
        the trajectory is issued with ``route=None`` and event type
        ``"plan_rejected"`` is logged instead.

        Parameters
        ----------
        drone_id:
            Drone node ID.
        role:
            Current role string (one of the ROLE_* constants).
        target_position:
            3-D position array of the assigned target, or ``None`` when the
            target is unknown / unassigned (e.g. ``ROLE_RESERVE``).
        drone_position:
            3-D position array of the drone at *timestamp_s*.
        planner:
            ``PathPlanner2D`` instance configured for the current scene.
        terrain_model:
            ``TerrainModel`` instance used to lift 2-D waypoints to 3-D.
        dynamics:
            ``DynamicsConfig`` instance supplying altitude and speed defaults.
        timestamp_s:
            Current simulation time, in seconds.
        """
        self._generation_counter += 1
        generation = self._generation_counter

        # Derive altitude profile from role + dynamics
        altitude_profile = self._altitude_profile_for_role(role, dynamics)

        # Choose speed from role
        speed_mps = self._speed_for_role(role, dynamics)

        # Build the 2-D goal position
        route = None
        override_reason: Optional[str] = None
        event_type = "plan_issued"

        drone_xy = np.asarray(drone_position, dtype=float)[:2]

        if target_position is not None and role not in (ROLE_RESERVE,):
            target_xy = np.asarray(target_position, dtype=float)[:2]
            goal_xy = self._goal_xy_for_role(role, drone_xy, target_xy, dynamics)

            # Attempt path planning
            try:
                route = planner.plan_route(  # type: ignore[union-attr]
                    drone_xy,
                    goal_xy,
                    clearance_m=getattr(getattr(planner, "config", None), "drone_clearance_m", 8.0),
                )
            except Exception:
                route = None

            if route is None:
                # Safety gate: plan rejected
                event_type = "plan_rejected"
                override_reason = "path_planner_returned_none"
        else:
            # Reserve drones or no-target situation: no route
            event_type = "plan_issued"

        # Compute staleness deadline
        route_length_m = float(route.length_m) if route is not None else 0.0  # type: ignore[union-attr]
        valid_until_s = self.compute_staleness(timestamp_s, route_length_m, speed_mps)

        trajectory = PlannedTrajectory(
            drone_id=drone_id,
            route=route,
            altitude_profile=altitude_profile,
            speed_mps=speed_mps,
            role=role,
            planned_at_s=timestamp_s,
            valid_until_s=valid_until_s,
            generation=generation,
            override_reason=override_reason,
        )

        self._plans[drone_id] = trajectory
        self._last_replan_time[drone_id] = timestamp_s

        self._record_event(
            PlannerEvent(
                timestamp_s=timestamp_s,
                drone_id=drone_id,
                event_type=event_type,
                trigger="plan_request",
                generation=generation,
                route_length_m=route_length_m if route is not None else None,
                override_reason=override_reason,
            )
        )

        return trajectory

    def should_replan(
        self,
        drone_id: str,
        timestamp_s: float,
        track_stale_steps: int = 0,
        obstacle_warning: bool = False,
        role_changed: bool = False,
    ) -> Tuple[bool, str]:
        """Decide whether *drone_id* should be replanned at *timestamp_s*.

        Evaluates replanning triggers in priority order (PLANNING.md §4.2):
        1. Track loss
        2. Obstacle ingress
        3. Role reassignment
        4. Staleness expiry

        Returns
        -------
        (should_replan, trigger_reason)
            ``True`` with a descriptive trigger string, or ``(False, "")``.
        """
        plan = self._plans.get(drone_id)
        last_time = self._last_replan_time.get(drone_id, -1e9)
        time_since_last = timestamp_s - last_time

        # --- Priority 1: Track loss ---
        # primary_observer and secondary_baseline use max_stale_steps=2 (§1)
        # generic default is 6 (DynamicsConfig.default_max_stale_steps)
        primary_roles = {ROLE_PRIMARY_OBSERVER, ROLE_SECONDARY_BASELINE}
        role = plan.role if plan is not None else ""
        max_stale = 2 if role in primary_roles else 6
        if track_stale_steps >= max_stale:
            trigger = "track_loss"
            if time_since_last >= self._replan_cooldown_s or trigger in _COOLDOWN_BYPASS_TRIGGERS:
                self._record_replan_trigger(drone_id, timestamp_s, trigger)
                return True, trigger

        # --- Priority 2: Obstacle ingress (bypasses cooldown) ---
        if obstacle_warning:
            trigger = "obstacle_ingress"
            self._record_replan_trigger(drone_id, timestamp_s, trigger)
            return True, trigger

        # --- Priority 3: Role reassignment (bypasses cooldown) ---
        if role_changed:
            trigger = "role_change"
            self._record_replan_trigger(drone_id, timestamp_s, trigger)
            return True, trigger

        # --- Cooldown gate for remaining triggers ---
        if time_since_last < self._replan_cooldown_s:
            return False, ""

        # --- Priority 4: Staleness expiry ---
        if plan is not None and timestamp_s >= plan.valid_until_s:
            trigger = "staleness_expiry"
            self._record_replan_trigger(drone_id, timestamp_s, trigger)
            return True, trigger

        return False, ""

    def check_deconfliction(
        self,
        drone_positions: Dict[str, np.ndarray],
        drone_roles: Dict[str, str],
        min_separation_m: float = 16.0,
    ) -> List[Tuple[str, str, float]]:
        """Evaluate separation constraints between all active (non-reserve) drones.

        Deconfliction is performed in 2-D XY space (PLANNING.md §5.1).

        Parameters
        ----------
        drone_positions:
            Mapping from drone_id to 3-D position array.
        drone_roles:
            Current role assignment for each drone.
        min_separation_m:
            Minimum required horizontal separation in metres
            (default 16 m = 2 × PlannerConfig.drone_clearance_m).

        Returns
        -------
        violations:
            List of ``(drone_a, drone_b, distance_m)`` tuples for every pair
            whose XY distance is below *min_separation_m*.  The lower-priority
            drone is listed second.  An empty list means no violations.
        """
        # Build list of active (non-reserve) drones with known positions
        active: List[Tuple[str, np.ndarray]] = []
        for drone_id, pos in drone_positions.items():
            role = drone_roles.get(drone_id, ROLE_RESERVE)
            if role != ROLE_RESERVE:
                active.append((drone_id, np.asarray(pos, dtype=float)))

        violations: List[Tuple[str, str, float]] = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                id_a, pos_a = active[i]
                id_b, pos_b = active[j]
                dist = float(np.linalg.norm(pos_a[:2] - pos_b[:2]))
                if dist < min_separation_m:
                    # Sort so higher-priority drone is first
                    pri_a = ROLE_PRIORITY.get(drone_roles.get(id_a, ROLE_RESERVE), 0)
                    pri_b = ROLE_PRIORITY.get(drone_roles.get(id_b, ROLE_RESERVE), 0)
                    if pri_a >= pri_b:
                        violations.append((id_a, id_b, dist))
                    else:
                        violations.append((id_b, id_a, dist))

        return violations

    def compute_staleness(
        self,
        planned_at_s: float,
        route_length_m: float,
        speed_mps: float,
    ) -> float:
        """Compute ``valid_until_s`` for a trajectory (PLANNING.md §4.1).

        ``valid_until_s = planned_at_s + max(30.0, route_length_m / speed_mps * 0.5)``
        """
        safe_speed = max(float(speed_mps), 1e-6)
        window_s = max(30.0, float(route_length_m) / safe_speed * 0.5)
        return float(planned_at_s) + window_s

    @property
    def events(self) -> List[PlannerEvent]:
        """Return a snapshot of all recorded planner events."""
        return list(self._events)

    @property
    def plans(self) -> Dict[str, PlannedTrajectory]:
        """Return a snapshot of the current plan per drone."""
        return dict(self._plans)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_event(self, event: PlannerEvent) -> None:
        self._events.append(event)

    def _record_replan_trigger(
        self, drone_id: str, timestamp_s: float, trigger: str
    ) -> None:
        plan = self._plans.get(drone_id)
        self._record_event(
            PlannerEvent(
                timestamp_s=timestamp_s,
                drone_id=drone_id,
                event_type="replan_trigger",
                trigger=trigger,
                generation=plan.generation if plan is not None else self._generation_counter,
            )
        )

    def _altitude_profile_for_role(
        self, role: str, dynamics: object
    ) -> AltitudeProfile:
        """Build an ``AltitudeProfile`` from *dynamics* suited to *role*."""
        # Pull attributes with safe fallbacks
        base_agl = float(getattr(dynamics, "drone_base_agl_m", 230.0))
        min_agl = float(getattr(dynamics, "follow_min_agl_m", 18.0))
        smoothing_s = float(getattr(dynamics, "terrain_following_smoothing_s", 1.5))

        if role == ROLE_PRIMARY_OBSERVER:
            base_agl = float(getattr(dynamics, "interceptor_follow_altitude_offset_m", 35.0))
            min_agl = float(getattr(dynamics, "interceptor_follow_min_agl_m", 150.0))
            mode = "fixed_agl"
        elif role == ROLE_SECONDARY_BASELINE:
            base_agl = float(getattr(dynamics, "tracker_altitude_offset_m", 70.0))
            min_agl = float(getattr(dynamics, "tracker_follow_min_agl_m", 155.0))
            mode = "fixed_agl"
        elif role == ROLE_CORRIDOR_WATCHER:
            mode = "terrain_following"
            min_agl = float(getattr(dynamics, "interceptor_search_min_agl_m", 180.0))
        elif role == ROLE_RELAY:
            mode = "fixed_agl"
            min_agl = float(getattr(dynamics, "interceptor_search_min_agl_m", 180.0))
        else:
            # reserve / unknown
            mode = "fixed_agl"

        max_agl = float(getattr(dynamics, "launch_operational_altitude_agl_m", 500.0))
        # Ensure base_agl is within [min_agl, max_agl]
        base_agl = max(base_agl, min_agl)
        base_agl = min(base_agl, max_agl)

        return AltitudeProfile(
            mode=mode,
            base_agl_m=base_agl,
            min_agl_m=min_agl,
            max_agl_m=max_agl,
            terrain_following_smoothing_s=smoothing_s,
        )

    def _speed_for_role(self, role: str, dynamics: object) -> float:
        """Return the nominal cruise speed in m/s for *role*."""
        if role in (ROLE_PRIMARY_OBSERVER, ROLE_SECONDARY_BASELINE):
            # Follow mode: cap at a reasonable intercept speed
            return float(getattr(dynamics, "drone_search_speed_base_mps", 28.0))
        if role == ROLE_CORRIDOR_WATCHER:
            return float(getattr(dynamics, "drone_search_speed_base_mps", 28.0))
        if role == ROLE_RELAY:
            # Relay loiters slowly
            return max(
                float(getattr(dynamics, "drone_search_speed_base_mps", 28.0)) * 0.3,
                8.0,
            )
        # reserve: not flying
        return float(getattr(dynamics, "drone_search_speed_base_mps", 28.0))

    def _goal_xy_for_role(
        self,
        role: str,
        drone_xy: np.ndarray,
        target_xy: np.ndarray,
        dynamics: object,
    ) -> np.ndarray:
        """Compute 2-D goal position based on role and target geometry."""
        if role == ROLE_PRIMARY_OBSERVER:
            standoff_m = float(getattr(dynamics, "interceptor_follow_radius_m", 55.0))
        elif role == ROLE_SECONDARY_BASELINE:
            standoff_m = float(getattr(dynamics, "tracker_standoff_radius_m", 120.0))
        elif role == ROLE_RELAY:
            # Relay: position midway between drone and target
            return (drone_xy + target_xy) * 0.5
        else:
            # corridor_watcher / default: move toward target
            standoff_m = 0.0

        if standoff_m > 0.0:
            direction = drone_xy - target_xy
            dist = float(np.linalg.norm(direction))
            if dist > 1e-6:
                direction = direction / dist
            else:
                direction = np.array([1.0, 0.0], dtype=float)
            return target_xy + direction * standoff_m

        return target_xy.copy()


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "ROLE_PRIMARY_OBSERVER",
    "ROLE_SECONDARY_BASELINE",
    "ROLE_CORRIDOR_WATCHER",
    "ROLE_RELAY",
    "ROLE_RESERVE",
    "ROLE_PRIORITY",
    "AltitudeProfile",
    "PlannedTrajectory",
    "PlanningObjectives",
    "PlannerEvent",
    "CooperativePlanner",
]
