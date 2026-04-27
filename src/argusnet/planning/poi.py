"""Point-of-Interest (POI) management for the inspection phase."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import InspectionPOI, POIStatus

# Travel power used by the battery model (watts).
_TRAVEL_POWER_W = 350.0


@dataclass(frozen=True)
class POIAssignmentContext:
    """Per-drone context for energy-aware POI assignment."""

    drone_id: str
    drone_pos: np.ndarray  # (3,) XYZ metres
    battery_remaining_wh: float
    battery_capacity_wh: float
    cruise_speed_mps: float = 28.0
    battery_reserve_fraction: float = 0.20
    timestamp_s: float = 0.0


class POIManager:
    """Tracks the lifecycle of all InspectionPOIs.

    Workflow:
      1. Initialise with a list of POIs.
      2. Call ``assign_nearest(drone_id, drone_pos, drone_battery_fraction)``
         to get the highest-priority unassigned POI within energy budget.
         OR call ``assign_energy_aware(context)`` for energy-budget-aware assignment.
      3. Call ``accumulate_dwell(drone_id, dt_s)`` each sim step while the
         drone is hovering at its assigned POI.
      4. Call ``check_completions(timestamp_s)`` each step; it returns IDs
         of newly completed POIs.
    """

    def __init__(self, pois: list[InspectionPOI]) -> None:
        self._pois: dict[str, InspectionPOI] = {p.poi_id: p for p in pois}
        self._statuses: dict[str, POIStatus] = {
            p.poi_id: POIStatus(poi_id=p.poi_id, status="pending") for p in pois
        }
        self._drone_assignment: dict[str, str] = {}  # drone_id -> poi_id
        self._dwell_acc: dict[str, float] = {}  # poi_id -> accumulated dwell
        self._effective_priorities: dict[str, int] = {p.poi_id: p.priority for p in pois}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def statuses(self) -> list[POIStatus]:
        return list(self._statuses.values())

    @property
    def completed_count(self) -> int:
        return sum(1 for s in self._statuses.values() if s.status == "complete")

    @property
    def total_count(self) -> int:
        return len(self._pois)

    @property
    def all_complete(self) -> bool:
        return self.completed_count == self.total_count

    def assignment_for(self, drone_id: str) -> str | None:
        """Return the poi_id currently assigned to this drone, or None."""
        return self._drone_assignment.get(drone_id)

    # ------------------------------------------------------------------
    # Assignment (original, kept for backward compatibility)
    # ------------------------------------------------------------------

    def assign_nearest(
        self,
        drone_id: str,
        drone_pos: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> InspectionPOI | None:
        """Assign the highest-priority nearby unassigned POI to drone_id.

        Already-assigned drones keep their current POI.
        Low-battery drones (< 25%) are not assigned a new POI.
        """
        # Already has one.
        if drone_id in self._drone_assignment:
            poi_id = self._drone_assignment[drone_id]
            if self._statuses[poi_id].status != "complete":
                return self._pois[poi_id]

        if battery_fraction < 0.25:
            return None

        pending = [
            poi for poi in self._pois.values() if self._statuses[poi.poi_id].status == "pending"
        ]
        if not pending:
            return None

        # Sort by priority desc, then distance asc.
        def sort_key(poi: InspectionPOI) -> tuple[int, float]:
            dist = float(np.linalg.norm(np.array(poi.position[:2]) - np.array(drone_pos[:2])))
            return (-poi.priority, dist)

        pending.sort(key=sort_key)
        chosen = pending[0]

        self._drone_assignment[drone_id] = chosen.poi_id
        self._dwell_acc[chosen.poi_id] = 0.0
        self._statuses[chosen.poi_id] = POIStatus(
            poi_id=chosen.poi_id,
            status="active",
            assigned_drone_id=drone_id,
        )
        return chosen

    # ------------------------------------------------------------------
    # Energy-aware assignment
    # ------------------------------------------------------------------

    def _energy_cost_wh(
        self,
        from_pos: np.ndarray,
        to_pos: np.ndarray,
        cruise_speed_mps: float,
    ) -> float:
        """Estimate energy to travel from *from_pos* to *to_pos* in Wh."""
        dist_m = float(np.linalg.norm(np.array(to_pos[:2]) - np.array(from_pos[:2])))
        travel_s = dist_m / max(cruise_speed_mps, 0.1)
        return _TRAVEL_POWER_W * travel_s / 3600.0

    def assign_energy_aware(self, context: POIAssignmentContext) -> InspectionPOI | None:
        """Assign a POI to a drone accounting for battery budget.

        Already-assigned drones keep their current POI.
        Drones that cannot afford the trip plus a reserve fraction are skipped.
        POIs are scored as: ``effective_priority * 10 - travel_cost / usable_wh``
        so energy is a tiebreaker within the same priority tier.
        """
        drone_id = context.drone_id
        # Keep existing assignment if still active.
        if drone_id in self._drone_assignment:
            poi_id = self._drone_assignment[drone_id]
            if self._statuses[poi_id].status != "complete":
                return self._pois[poi_id]

        usable_wh = context.battery_remaining_wh - (
            context.battery_capacity_wh * context.battery_reserve_fraction
        )
        if usable_wh <= 0.0:
            return None

        pending = [
            poi for poi in self._pois.values() if self._statuses[poi.poi_id].status == "pending"
        ]
        if not pending:
            return None

        best_poi: InspectionPOI | None = None
        best_score = float("-inf")
        for poi in pending:
            cost_wh = self._energy_cost_wh(
                context.drone_pos, poi.position, context.cruise_speed_mps
            )
            if cost_wh > usable_wh * 0.8:
                # Not enough energy to reach this POI and keep reserve.
                continue
            eff_priority = self._effective_priorities.get(poi.poi_id, poi.priority)
            score = eff_priority * 10.0 - cost_wh / max(usable_wh, 0.01)
            if score > best_score:
                best_score = score
                best_poi = poi

        if best_poi is None:
            return None

        self._drone_assignment[drone_id] = best_poi.poi_id
        self._dwell_acc[best_poi.poi_id] = 0.0
        self._statuses[best_poi.poi_id] = POIStatus(
            poi_id=best_poi.poi_id,
            status="active",
            assigned_drone_id=drone_id,
        )
        return best_poi

    # ------------------------------------------------------------------
    # Handoff and team assignment
    # ------------------------------------------------------------------

    def trigger_handoff(self, from_drone_id: str, to_drone_id: str) -> bool:
        """Reassign an active POI from *from_drone_id* to *to_drone_id*.

        Returns True if the handoff succeeded, False otherwise.
        """
        poi_id = self._drone_assignment.get(from_drone_id)
        if poi_id is None:
            return False
        st = self._statuses[poi_id]
        if st.status != "active":
            return False
        # Release from old drone.
        del self._drone_assignment[from_drone_id]
        # Assign to new drone.
        self._drone_assignment[to_drone_id] = poi_id
        from dataclasses import replace

        self._statuses[poi_id] = replace(st, assigned_drone_id=to_drone_id)
        return True

    def request_team_assign(self, poi_id: str, second_drone_id: str) -> bool:
        """Add a second drone to help dwell on an already-active POI.

        Both drones' ``accumulate_dwell`` calls credit the same POI, halving
        the effective dwell time needed.  Returns True if successful.
        """
        st = self._statuses.get(poi_id)
        if st is None or st.status != "active":
            return False
        if second_drone_id in self._drone_assignment:
            return False
        self._drone_assignment[second_drone_id] = poi_id
        return True

    def rescore_from_map(self, world_map) -> None:
        """Adjust effective priorities based on local coverage density.

        POIs in poorly-covered regions get a priority boost (up to +3) so that
        inspection effort follows coverage gaps in the scan map.
        """
        for poi_id, poi in self._pois.items():
            if self._statuses[poi_id].status != "pending":
                continue
            try:
                density = world_map.coverage_in_region(poi.position[:2], radius_m=50.0)
            except Exception:
                density = 1.0
            boost = int(round((1.0 - float(density)) * 3))
            self._effective_priorities[poi_id] = poi.priority + boost

    # ------------------------------------------------------------------
    # Dwell accumulation
    # ------------------------------------------------------------------

    def accumulate_dwell(self, drone_id: str, drone_pos: np.ndarray, dt_s: float) -> None:
        """Credit dwell time if the drone is within 30 m of its assigned POI."""
        poi_id = self._drone_assignment.get(drone_id)
        if poi_id is None:
            return
        poi = self._pois[poi_id]
        dist = float(np.linalg.norm(np.array(poi.position[:2]) - np.array(drone_pos[:2])))
        if dist <= 30.0:
            self._dwell_acc[poi_id] = self._dwell_acc.get(poi_id, 0.0) + dt_s

    def check_completions(self, timestamp_s: float) -> list[str]:
        """Return list of poi_ids that just became complete this step."""
        newly_done: list[str] = []
        for poi_id, poi in self._pois.items():
            st = self._statuses[poi_id]
            if st.status != "active":
                continue
            dwell = self._dwell_acc.get(poi_id, 0.0)
            if dwell >= poi.required_dwell_s:
                arr_time = st.arrival_time_s or timestamp_s
                self._statuses[poi_id] = POIStatus(
                    poi_id=poi_id,
                    status="complete",
                    assigned_drone_id=st.assigned_drone_id,
                    arrival_time_s=arr_time,
                    completion_time_s=timestamp_s,
                    dwell_accumulated_s=dwell,
                )
                # Release ALL drones assigned to this POI (supports team assign).
                for did in list(self._drone_assignment.keys()):
                    if self._drone_assignment[did] == poi_id:
                        del self._drone_assignment[did]
                newly_done.append(poi_id)
        return newly_done
