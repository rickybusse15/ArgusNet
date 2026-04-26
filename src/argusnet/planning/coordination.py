"""Multi-drone coordination for ISR missions.

Provides:
  - Coordinator election (highest-battery drone becomes coordinator).
  - RF-latency simulation for shared claimed-cell updates.
  - Formation offset computation for scanning phase (line-abreast, V-formation).

The mutable ``SharedMissionState`` is owned by the sim loop and passed to
``CoordinationManager`` methods rather than stored inside the manager.  This
keeps the manager stateless (aside from the policy config) and makes it easy
to test in isolation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = ["CoordinationPolicy", "SharedMissionState", "CoordinationManager"]


@dataclass(frozen=True)
class CoordinationPolicy:
    """Configuration for multi-drone coordination behaviour."""
    elect_coordinator: bool = True
    # If True, the drone with the highest remaining battery is designated
    # coordinator each time election is requested.
    message_latency_steps: int = 0
    # Simulated RF communication latency in sim steps.
    # 0 = instant delivery (current behaviour, backward compatible).
    formation_mode: str = "none"
    # "none"         — no formation; each drone picks its own frontier target.
    # "line_abreast" — drones fly abreast perpendicular to lead heading.
    # "v_formation"  — drones form a V behind the lead drone.
    formation_spacing_m: float = 80.0
    # Lateral spacing between adjacent drones in the formation.


@dataclass
class SharedMissionState:
    """Mutable coordination state shared across all drone planners each step.

    NOT frozen (mutated each step) and NOT serialised into replay JSON.
    The serialisable fields (coordinator_drone_id, etc.) are written into
    ScanMissionState separately.
    """
    claimed_cells: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # drone_id -> (ci, cj) grid cell the drone is currently heading toward.
    poi_assignments: Dict[str, str] = field(default_factory=dict)
    # drone_id -> poi_id tentative assignment (informational).
    coordinator_id: Optional[str] = None
    # The currently elected coordinator drone_id.
    pending_messages: Dict[str, List[dict]] = field(
        default_factory=lambda: defaultdict(list)
    )
    # Queued claimed-cell updates awaiting delivery (RF latency simulation).
    # Each entry: {"drone_id": str, "cell": (ci, cj), "deliver_at_step": int}
    message_latency_steps: int = 0


class CoordinationManager:
    """Implements multi-drone coordination primitives."""

    def __init__(self, policy: Optional[CoordinationPolicy] = None) -> None:
        self.policy = policy or CoordinationPolicy()

    # ------------------------------------------------------------------
    # Coordinator election
    # ------------------------------------------------------------------

    def elect_coordinator(
        self,
        drone_ids: List[str],
        battery_states: Dict[str, object],   # drone_id -> BatteryState
        battery_capacity_wh: float = 500.0,
    ) -> Optional[str]:
        """Return the drone_id with the highest remaining battery fraction.

        The result is deterministic for a given set of battery states.
        Returns None if *drone_ids* is empty.
        """
        if not drone_ids:
            return None

        best_id = drone_ids[0]
        best_frac = -1.0
        for did in drone_ids:
            bs = battery_states.get(did)
            frac = bs.remaining_wh / max(battery_capacity_wh, 1.0) if bs else 1.0
            if frac > best_frac:
                best_frac = frac
                best_id = did
        return best_id

    # ------------------------------------------------------------------
    # Claimed-cell updates with optional RF latency
    # ------------------------------------------------------------------

    def update_claimed(
        self,
        drone_id: str,
        cell: Optional[Tuple[int, int]],
        shared_state: SharedMissionState,
        current_step: int,
    ) -> None:
        """Record a claimed-cell update, respecting simulated message latency.

        If ``message_latency_steps == 0`` the update is applied immediately
        (preserving the original behaviour).  Otherwise, it is queued and
        will be applied by ``flush_messages()`` once enough steps have passed.
        """
        if cell is None:
            return

        latency = shared_state.message_latency_steps
        if latency == 0:
            shared_state.claimed_cells[drone_id] = cell
        else:
            deliver_at = current_step + latency
            shared_state.pending_messages[drone_id].append({
                "cell": cell,
                "deliver_at_step": deliver_at,
            })

    def flush_messages(
        self, shared_state: SharedMissionState, current_step: int
    ) -> None:
        """Apply queued messages whose delivery step has been reached."""
        for drone_id, messages in list(shared_state.pending_messages.items()):
            remaining = []
            for msg in messages:
                if msg["deliver_at_step"] <= current_step:
                    shared_state.claimed_cells[drone_id] = msg["cell"]
                else:
                    remaining.append(msg)
            shared_state.pending_messages[drone_id] = remaining

    # ------------------------------------------------------------------
    # Formation offsets
    # ------------------------------------------------------------------

    def formation_offsets(
        self,
        drone_ids: List[str],
        lead_heading_rad: float,
    ) -> Dict[str, np.ndarray]:
        """Compute per-drone XY offset to add to each drone's target waypoint.

        Returns an empty dict when ``formation_mode == "none"`` or fewer than
        2 drones are present.

        Formation conventions:
          - Drone at index 0 is the lead (offset = zero).
          - ``line_abreast``: all drones fly side-by-side, perpendicular to
            the lead heading.  Drone 1 is to the left, drone 2 to the right,
            drone 3 further left, etc.
          - ``v_formation``: drones trail in a V behind the lead.  Drone 1 is
            behind-left, drone 2 behind-right, alternating.
        """
        policy = self.policy
        if policy.formation_mode == "none" or len(drone_ids) < 2:
            return {}

        s = policy.formation_spacing_m
        cos_h = math.cos(lead_heading_rad)
        sin_h = math.sin(lead_heading_rad)

        # Perpendicular (left) unit vector in the XY plane.
        perp = np.array([-sin_h, cos_h], dtype=float)
        # Along-track unit vector (forward).
        fwd = np.array([cos_h, sin_h], dtype=float)

        offsets: Dict[str, np.ndarray] = {}
        for idx, did in enumerate(drone_ids):
            if idx == 0:
                offsets[did] = np.zeros(2, dtype=float)
                continue

            if policy.formation_mode == "line_abreast":
                # Alternate left/right: +1, −1, +2, −2, …
                sign = 1 if idx % 2 == 1 else -1
                rank = (idx + 1) // 2
                offsets[did] = perp * (sign * rank * s)

            elif policy.formation_mode == "v_formation":
                # Trail behind the lead, alternating left/right.
                sign = 1 if idx % 2 == 1 else -1
                rank = (idx + 1) // 2
                offsets[did] = -fwd * (rank * s * 0.5) + perp * (sign * rank * s * 0.5)
            else:
                offsets[did] = np.zeros(2, dtype=float)

        return offsets
