"""Battery-aware route budget and return-to-home planning for ArgusNet.

Provides a simple energy model and waypoint insertion to ensure a drone
can always return to its home position with a minimum reserve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3

__all__ = [
    "BatteryModel",
    "BatteryState",
    "BudgetResult",
    "compute_route_budget",
    "insert_return_waypoint",
]


@dataclass(frozen=True)
class BatteryModel:
    """Energy consumption model for a multirotor drone.

    Consumption is modelled as a constant rate (W) divided into
    horizontal travel, vertical travel, and hover components.
    """

    capacity_wh: float = 100.0
    """Battery capacity in watt-hours."""

    hover_power_w: float = 300.0
    """Power draw while hovering (W)."""

    travel_power_w: float = 350.0
    """Power draw during horizontal cruise (W)."""

    climb_power_w: float = 500.0
    """Power draw during climb (W)."""

    cruise_speed_m_per_s: float = 10.0
    """Nominal cruise speed (m/s)."""

    climb_speed_m_per_s: float = 3.0
    """Nominal climb/descent speed (m/s)."""

    reserve_fraction: float = 0.20
    """Minimum battery fraction to retain at home (safety reserve)."""

    def travel_cost_wh(self, distance_m: float) -> float:
        """Energy cost (Wh) for a horizontal segment of *distance_m*."""
        duration_s = distance_m / max(self.cruise_speed_m_per_s, 0.01)
        return self.travel_power_w * duration_s / 3600.0

    def climb_cost_wh(self, delta_z_m: float) -> float:
        """Energy cost (Wh) for a vertical segment of *delta_z_m* metres."""
        duration_s = abs(delta_z_m) / max(self.climb_speed_m_per_s, 0.01)
        power = self.climb_power_w if delta_z_m > 0 else self.hover_power_w * 0.8
        return power * duration_s / 3600.0

    def segment_cost_wh(self, start: Vector3, end: Vector3) -> float:
        """Total energy cost for moving from *start* to *end*."""
        start, end = np.asarray(start), np.asarray(end)
        h_dist = float(np.linalg.norm(end[:2] - start[:2]))
        dz = float(end[2] - start[2])
        return self.travel_cost_wh(h_dist) + self.climb_cost_wh(dz)

    def route_cost_wh(self, waypoints: list[Vector3]) -> float:
        """Total energy cost for a sequence of waypoints."""
        total = 0.0
        for a, b in zip(waypoints[:-1], waypoints[1:], strict=False):
            total += self.segment_cost_wh(a, b)
        return total

    @property
    def usable_capacity_wh(self) -> float:
        return self.capacity_wh * (1.0 - self.reserve_fraction)


@dataclass
class BatteryState:
    """Current battery state during a mission."""

    remaining_wh: float
    timestamp_s: float

    @property
    def fraction(self) -> float:
        return self.remaining_wh  # caller normalises against capacity

    def consume(self, cost_wh: float) -> BatteryState:
        return BatteryState(
            remaining_wh=max(0.0, self.remaining_wh - cost_wh),
            timestamp_s=self.timestamp_s,
        )


@dataclass(frozen=True)
class BudgetResult:
    """Result of a route budget analysis."""

    route_cost_wh: float
    """Energy required to fly the entire planned route."""

    return_cost_wh: float
    """Energy required to return from current position to home."""

    reserve_wh: float
    """Minimum safe reserve to keep."""

    available_wh: float
    """Energy available for the mission (capacity - reserve)."""

    feasible: bool
    """True if route_cost_wh + return_cost_wh <= available_wh."""

    max_safe_segments: int
    """How many planned segments can be flown before must-return."""


def compute_route_budget(
    waypoints: list[Vector3],
    home: Vector3,
    model: BatteryModel,
    current_wh: float | None = None,
) -> BudgetResult:
    """Analyse whether the remaining battery supports flying *waypoints*.

    Args:
        waypoints: Planned waypoints (starting from current position).
        home: Home/landing position.
        model: Battery model.
        current_wh: Current battery energy (Wh). Defaults to full capacity.

    Returns:
        A :class:`BudgetResult` describing feasibility.
    """
    available = current_wh if current_wh is not None else model.capacity_wh
    reserve = model.capacity_wh * model.reserve_fraction
    usable = available - reserve

    if not waypoints:
        return_cost = model.segment_cost_wh(home, home)
        return BudgetResult(
            route_cost_wh=0.0,
            return_cost_wh=return_cost,
            reserve_wh=reserve,
            available_wh=usable,
            feasible=True,
            max_safe_segments=0,
        )

    # Energy from each segment onwards (suffix sums)
    seg_costs = [
        model.segment_cost_wh(waypoints[k], waypoints[k + 1]) for k in range(len(waypoints) - 1)
    ]
    # Return cost from each waypoint to home
    return_costs = [model.segment_cost_wh(wp, home) for wp in waypoints]

    total_route = sum(seg_costs)
    return_from_end = return_costs[-1]

    # Find how many segments we can do before must-return
    cumulative = 0.0
    max_segs = 0
    for k, (sc, rc) in enumerate(zip(seg_costs, return_costs[1:], strict=False)):
        if cumulative + sc + rc <= usable:
            cumulative += sc
            max_segs = k + 1
        else:
            break

    feasible = (total_route + return_from_end) <= usable

    return BudgetResult(
        route_cost_wh=total_route,
        return_cost_wh=return_from_end,
        reserve_wh=reserve,
        available_wh=usable,
        feasible=feasible,
        max_safe_segments=max_segs,
    )


def insert_return_waypoint(
    waypoints: list[Vector3],
    home: Vector3,
    model: BatteryModel,
    current_wh: float,
) -> list[Vector3]:
    """Truncate *waypoints* at the energy budget limit and append *home*.

    Returns the safe sub-route ending at *home*.
    """
    budget = compute_route_budget(waypoints, home, model, current_wh)
    safe = list(waypoints[: budget.max_safe_segments + 1])
    if not safe:
        safe = [waypoints[0]] if waypoints else [home]
    safe.append(home)
    return safe
