from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from argusnet.core.types import DeconflictionEvent, MissionZone, Vector3

if TYPE_CHECKING:
    from argusnet.planning.inspection import FlightCorridor
    from argusnet.world.terrain import TerrainModel

__all__ = [
    "DEFAULT_ROLE_PRIORITY",
    "CorridorWindowAllocator",
    "DeconflictionConfig",
    "DeconflictionEvent",
    "DeconflictionLayer",
    "ROLE_PRIORITY",
]


ROLE_PRIORITY: dict[str, int] = {
    "primary_observer": 5,
    "secondary_baseline": 4,
    "corridor_watcher": 3,
    "relay": 2,
    "reserve": 1,
}

DEFAULT_ROLE_PRIORITY = 0


@dataclass(frozen=True)
class DeconflictionConfig:
    min_separation_m: float = 16.0
    look_ahead_s: float = 1.0
    vertical_separation_m: float = 20.0
    min_agl_m: float = 18.0
    corridor_window_default_s: float = 5.0
    enabled: bool = True


@dataclass(frozen=True)
class _CorridorFrame:
    direction_xy: np.ndarray
    distance_m: float


@dataclass(frozen=True)
class _CorridorHoldDecision:
    holding_drone_id: str
    conflicting_drone_id: str
    predicted_separation_m: float
    corridor_id: str


def _priority(role: str) -> int:
    return ROLE_PRIORITY.get(role, DEFAULT_ROLE_PRIORITY)


def _yielder(a_id: str, a_role: str, b_id: str, b_role: str) -> tuple[str, str]:
    """Return (yielder_id, peer_id). Lower priority yields; ties broken by lex id."""
    pa, pb = _priority(a_role), _priority(b_role)
    if pa < pb:
        return a_id, b_id
    if pb < pa:
        return b_id, a_id
    return (a_id, b_id) if a_id > b_id else (b_id, a_id)


class CorridorWindowAllocator:
    """Alternating HOLD/GO allocator for bidirectional flight corridors.

    The open travel direction alternates every corridor window. Window length is
    corridor length divided by observed drone speed when available; otherwise
    ``default_window_s`` is used. Direction is measured along the corridor
    first-to-last waypoint axis.
    """

    def __init__(
        self,
        corridors: Sequence[FlightCorridor],
        role_lookup: Mapping[str, str],
        *,
        default_window_s: float = 5.0,
    ) -> None:
        self._corridors = tuple(corridors)
        self._roles = dict(role_lookup)
        self._default_window_s = max(float(default_window_s), 1.0e-6)

    def hold_decisions(
        self,
        proposed: Mapping[str, tuple[Vector3, Vector3]],
        timestamp_s: float,
    ) -> list[_CorridorHoldDecision]:
        decisions: list[_CorridorHoldDecision] = []
        held: set[str] = set()

        for corridor in self._corridors:
            if getattr(corridor, "direction", "bidirectional") != "bidirectional":
                continue
            if not self._is_active(corridor, timestamp_s):
                continue

            members = self._corridor_members(corridor, proposed)
            if len(members) < 2:
                continue

            window_s = self._window_duration_s(corridor, members)
            open_sign = self._open_direction_sign(corridor, timestamp_s, window_s)

            for i, (a_id, a_sign, _, a_pos) in enumerate(members):
                for b_id, b_sign, _, b_pos in members[i + 1 :]:
                    if a_id in held or b_id in held or a_sign == b_sign:
                        continue

                    if a_sign != open_sign and b_sign == open_sign:
                        holding_id, peer_id = a_id, b_id
                    elif b_sign != open_sign and a_sign == open_sign:
                        holding_id, peer_id = b_id, a_id
                    else:
                        holding_id, peer_id = _yielder(
                            a_id, self._roles.get(a_id, ""), b_id, self._roles.get(b_id, "")
                        )

                    held.add(holding_id)
                    sep = float(np.linalg.norm(a_pos[:2] - b_pos[:2]))
                    decisions.append(
                        _CorridorHoldDecision(
                            holding_drone_id=holding_id,
                            conflicting_drone_id=peer_id,
                            predicted_separation_m=sep,
                            corridor_id=str(getattr(corridor, "corridor_id", "")),
                        )
                    )

        return decisions

    def _corridor_members(
        self,
        corridor: FlightCorridor,
        proposed: Mapping[str, tuple[Vector3, Vector3]],
    ) -> list[tuple[str, int, float, Vector3]]:
        width_m = float(getattr(corridor, "width_m", 0.0))
        assigned = set(getattr(corridor, "assigned_drone_ids", []) or [])
        members: list[tuple[str, int, float, Vector3]] = []
        for drone_id, (pos, vel) in proposed.items():
            if assigned and drone_id not in assigned:
                continue
            frame = self._corridor_frame(corridor, pos[:2])
            if frame is None or frame.distance_m > width_m:
                continue
            speed_along = float(np.dot(np.asarray(vel[:2], dtype=float), frame.direction_xy))
            if abs(speed_along) <= 1.0e-6:
                continue
            sign = 1 if speed_along > 0.0 else -1
            members.append((drone_id, sign, abs(speed_along), np.asarray(pos, dtype=float)))
        return members

    def _window_duration_s(
        self,
        corridor: FlightCorridor,
        members: Sequence[tuple[str, int, float, Vector3]],
    ) -> float:
        length_m = self._corridor_length_m(corridor)
        speeds = [speed for _, _, speed, _ in members if speed > 1.0e-6]
        if length_m > 0.0 and speeds:
            return max(length_m / (sum(speeds) / len(speeds)), 1.0e-6)
        return self._default_window_s

    @staticmethod
    def _open_direction_sign(
        corridor: FlightCorridor,
        timestamp_s: float,
        window_s: float,
    ) -> int:
        active_window = getattr(corridor, "active_window", [0.0, float("inf")])
        start_s = float(active_window[0]) if active_window else 0.0
        window_index = int(max(0.0, timestamp_s - start_s) // max(window_s, 1.0e-6))
        return 1 if window_index % 2 == 0 else -1

    @staticmethod
    def _is_active(corridor: FlightCorridor, timestamp_s: float) -> bool:
        active_window = getattr(corridor, "active_window", [0.0, float("inf")])
        if len(active_window) < 2:
            return True
        return float(active_window[0]) <= timestamp_s <= float(active_window[1])

    @staticmethod
    def _corridor_length_m(corridor: FlightCorridor) -> float:
        points = np.asarray(getattr(corridor, "waypoints_xy_m", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1).sum())

    @staticmethod
    def _corridor_frame(corridor: FlightCorridor, xy: Vector3) -> _CorridorFrame | None:
        points = np.asarray(getattr(corridor, "waypoints_xy_m", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return None

        point_xy = np.asarray(xy[:2], dtype=float)
        best: _CorridorFrame | None = None
        for start, end in zip(points[:-1, :2], points[1:, :2], strict=False):
            segment = end - start
            length = float(np.linalg.norm(segment))
            if length <= 1.0e-9:
                continue
            direction = segment / length
            along = float(np.dot(point_xy - start, direction))
            closest = start + direction * float(np.clip(along, 0.0, length))
            distance_m = float(np.linalg.norm(point_xy - closest))
            if best is None or distance_m < best.distance_m:
                best = _CorridorFrame(direction_xy=direction, distance_m=distance_m)
        return best


class DeconflictionLayer:
    """Per-step pairwise separation enforcement for multi-drone scenarios.

    Runs after controllers have produced proposed (pos, vel) for each drone.
    Corridor windows are applied first by zeroing a held drone velocity. Pairwise
    separation then resolves close approaches through lateral or vertical yields.
    """

    def __init__(
        self,
        config: DeconflictionConfig,
        role_lookup: Mapping[str, str],
        terrain: TerrainModel | None = None,
        exclusion_zones: Sequence[MissionZone] | None = None,
        corridors: Sequence[FlightCorridor] | None = None,
    ) -> None:
        self.config = config
        self._roles = dict(role_lookup)
        self._terrain = terrain
        self._exclusion_zones: list[MissionZone] = [
            z for z in (exclusion_zones or []) if z.zone_type == "exclusion"
        ]
        self._corridor_allocator = CorridorWindowAllocator(
            corridors or (),
            role_lookup,
            default_window_s=config.corridor_window_default_s,
        )

    def resolve_step(
        self,
        proposed: Mapping[str, tuple[Vector3, Vector3]],
        timestamp_s: float,
    ) -> tuple[dict[str, tuple[Vector3, Vector3]], list[DeconflictionEvent]]:
        adjusted: dict[str, tuple[Vector3, Vector3]] = {
            d: (np.array(p, dtype=float), np.array(v, dtype=float))
            for d, (p, v) in proposed.items()
        }
        events: list[DeconflictionEvent] = []

        if not self.config.enabled or len(adjusted) < 2:
            return adjusted, events

        for hold in self._corridor_allocator.hold_decisions(adjusted, timestamp_s):
            pos, vel = adjusted[hold.holding_drone_id]
            adjusted[hold.holding_drone_id] = (pos, np.zeros_like(vel))
            events.append(
                DeconflictionEvent(
                    yielding_drone_id=hold.holding_drone_id,
                    conflicting_drone_id=hold.conflicting_drone_id,
                    predicted_separation_m=hold.predicted_separation_m,
                    resolution="corridor_hold",
                    timestamp_s=float(timestamp_s),
                )
            )

        ids = sorted(adjusted.keys())
        look = float(self.config.look_ahead_s)
        min_sep = float(self.config.min_separation_m)
        vert_sep = float(self.config.vertical_separation_m)

        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1 :]:
                a_pos, a_vel = adjusted[a_id]
                b_pos, b_vel = adjusted[b_id]

                pred_sep = self._closest_approach_xy(a_pos, a_vel, b_pos, b_vel, look)
                if pred_sep >= min_sep:
                    continue

                yielder_id, peer_id = _yielder(
                    a_id, self._roles.get(a_id, ""), b_id, self._roles.get(b_id, "")
                )
                y_pos, y_vel = adjusted[yielder_id]
                p_pos, _ = adjusted[peer_id]

                push_xy = self._lateral_push(y_pos, y_vel, p_pos, min_sep, pred_sep)
                lat_candidate = y_pos + np.array([push_xy[0], push_xy[1], 0.0], dtype=float)

                new_pred_xy = lat_candidate[:2] + y_vel[:2] * look
                still_close = float(np.linalg.norm(new_pred_xy - p_pos[:2])) < min_sep
                lateral_in_zone = self._in_exclusion_zone(lat_candidate[:2])

                if not still_close and not lateral_in_zone:
                    new_pos = lat_candidate
                    resolution = "lateral_offset"
                else:
                    opp_candidate = y_pos - np.array([push_xy[0], push_xy[1], 0.0], dtype=float)
                    opp_pred_xy = opp_candidate[:2] + y_vel[:2] * look
                    opp_close = float(np.linalg.norm(opp_pred_xy - p_pos[:2])) < min_sep
                    opp_in_zone = self._in_exclusion_zone(opp_candidate[:2])
                    if not opp_close and not opp_in_zone:
                        new_pos = opp_candidate
                        resolution = "lateral_offset"
                    else:
                        direction = 1.0 if y_pos[2] >= p_pos[2] else -1.0
                        new_pos = y_pos + np.array([0.0, 0.0, direction * vert_sep], dtype=float)
                        resolution = "vertical_offset"

                new_pos = self._clamp_terrain(new_pos)

                adjusted[yielder_id] = (new_pos, y_vel)
                events.append(
                    DeconflictionEvent(
                        yielding_drone_id=yielder_id,
                        conflicting_drone_id=peer_id,
                        predicted_separation_m=pred_sep,
                        resolution=resolution,
                        timestamp_s=float(timestamp_s),
                    )
                )

        return adjusted, events

    def _clamp_terrain(self, pos: np.ndarray) -> np.ndarray:
        """Clamp Z so the drone never goes below terrain + min_agl_m."""
        if self._terrain is None:
            return pos
        return np.array(
            [
                float(pos[0]),
                float(pos[1]),
                self._terrain.clamp_altitude(pos[:2], float(pos[2]), self.config.min_agl_m),
            ],
            dtype=float,
        )

    def _in_exclusion_zone(self, xy: np.ndarray) -> bool:
        """Return True if xy falls inside any exclusion zone footprint."""
        return any(zone.contains_point(xy) for zone in self._exclusion_zones)

    @staticmethod
    def _closest_approach_xy(
        a_pos: Vector3,
        a_vel: Vector3,
        b_pos: Vector3,
        b_vel: Vector3,
        look_ahead_s: float,
    ) -> float:
        """Minimum XY distance between two drones over [0, look_ahead_s]."""
        r0 = a_pos[:2] - b_pos[:2]
        vr = a_vel[:2] - b_vel[:2]
        speed_sq = float(np.dot(vr, vr))
        if speed_sq <= 1e-12:
            t_star = 0.0
        else:
            t_star = -float(np.dot(r0, vr)) / speed_sq
            t_star = max(0.0, min(t_star, float(look_ahead_s)))
        sep_vec = r0 + vr * t_star
        return float(np.linalg.norm(sep_vec))

    @staticmethod
    def _lateral_push(
        y_pos: Vector3,
        y_vel: Vector3,
        peer_pos: Vector3,
        min_sep: float,
        pred_sep: float,
    ) -> np.ndarray:
        """Pick a perpendicular-to-heading offset that pushes away from the peer."""
        margin = 1.0
        push_mag = max(min_sep - pred_sep + margin, margin)

        speed_xy = float(np.linalg.norm(y_vel[:2]))
        if speed_xy > 1e-6:
            heading = y_vel[:2] / speed_xy
            perp = np.array([-heading[1], heading[0]], dtype=float)
        else:
            away = y_pos[:2] - peer_pos[:2]
            n = float(np.linalg.norm(away))
            perp = away / n if n > 1e-6 else np.array([1.0, 0.0], dtype=float)

        peer_dir = peer_pos[:2] - y_pos[:2]
        if float(np.dot(perp, peer_dir)) > 0.0:
            perp = -perp

        return perp * push_mag
