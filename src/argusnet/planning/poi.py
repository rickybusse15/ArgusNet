"""Point-of-Interest (POI) management for the inspection phase."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from argusnet.core.types import InspectionPOI, POIStatus


class POIManager:
    """Tracks the lifecycle of all InspectionPOIs.

    Workflow:
      1. Initialise with a list of POIs.
      2. Call ``assign_nearest(drone_id, drone_pos, drone_battery_fraction)``
         to get the highest-priority unassigned POI within energy budget.
      3. Call ``accumulate_dwell(drone_id, dt_s)`` each sim step while the
         drone is hovering at its assigned POI.
      4. Call ``check_completions(timestamp_s)`` each step; it returns IDs
         of newly completed POIs.
    """

    def __init__(self, pois: List[InspectionPOI]) -> None:
        self._pois: Dict[str, InspectionPOI] = {p.poi_id: p for p in pois}
        self._statuses: Dict[str, POIStatus] = {
            p.poi_id: POIStatus(poi_id=p.poi_id, status="pending")
            for p in pois
        }
        self._drone_assignment: Dict[str, str] = {}   # drone_id -> poi_id
        self._dwell_acc: Dict[str, float] = {}         # poi_id -> accumulated dwell

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def statuses(self) -> List[POIStatus]:
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

    def assignment_for(self, drone_id: str) -> Optional[str]:
        """Return the poi_id currently assigned to this drone, or None."""
        return self._drone_assignment.get(drone_id)

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def assign_nearest(
        self,
        drone_id: str,
        drone_pos: np.ndarray,
        battery_fraction: float = 1.0,
    ) -> Optional[InspectionPOI]:
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
            poi for poi in self._pois.values()
            if self._statuses[poi.poi_id].status == "pending"
        ]
        if not pending:
            return None

        # Sort by priority desc, then distance asc.
        def sort_key(poi: InspectionPOI) -> Tuple[int, float]:
            dist = float(np.linalg.norm(
                np.array(poi.position[:2]) - np.array(drone_pos[:2])
            ))
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
    # Dwell accumulation
    # ------------------------------------------------------------------

    def accumulate_dwell(self, drone_id: str, drone_pos: np.ndarray, dt_s: float) -> None:
        """Credit dwell time if the drone is within 30 m of its assigned POI."""
        poi_id = self._drone_assignment.get(drone_id)
        if poi_id is None:
            return
        poi = self._pois[poi_id]
        dist = float(np.linalg.norm(
            np.array(poi.position[:2]) - np.array(drone_pos[:2])
        ))
        if dist <= 30.0:
            self._dwell_acc[poi_id] = self._dwell_acc.get(poi_id, 0.0) + dt_s

    def check_completions(self, timestamp_s: float) -> List[str]:
        """Return list of poi_ids that just became complete this step."""
        newly_done: List[str] = []
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
                drone_id = st.assigned_drone_id
                if drone_id and drone_id in self._drone_assignment:
                    del self._drone_assignment[drone_id]
                newly_done.append(poi_id)
        return newly_done
