"""MAVLink / telemetry stream ingestion for ArgusNet.

Parses common MAVLink message types (or JSON equivalents) into the
:class:`~argusnet.core.types.NodeState` and related structures used by
the ArgusNet ingestion pipeline.  Works in two modes:

1. **Live serial / UDP** — wraps :pypi:`pymavlink` to read from an
   autopilot connection (serial, UDP, TCP).
2. **JSON replay** — reads a JSON-lines file of pre-extracted telemetry
   messages (useful for offline testing without pymavlink installed).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from argusnet.core.frames import ENUOrigin, wgs84_to_enu
from argusnet.core.types import NodeState, vec3

logger = logging.getLogger(__name__)

__all__ = [
    "TelemetryMessage",
    "parse_gps_raw_int",
    "parse_attitude",
    "parse_battery_status",
    "parse_heartbeat",
    "TelemetryAggregator",
    "JSONTelemetryReplay",
]


# ---------------------------------------------------------------------------
# Telemetry message container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryMessage:
    """Unified container for a single telemetry datum."""

    msg_type: str
    """MAVLink message name (e.g. ``GPS_RAW_INT``, ``ATTITUDE``)."""

    node_id: str
    """Identifier of the originating vehicle / system."""

    timestamp_s: float
    """Timestamp in simulation or UNIX seconds."""

    fields: dict[str, Any] = field(default_factory=dict)
    """Message-specific fields."""


# ---------------------------------------------------------------------------
# MAVLink message parsers
# ---------------------------------------------------------------------------


def parse_gps_raw_int(
    msg: dict[str, Any],
    node_id: str,
    enu_origin: ENUOrigin | None = None,
) -> TelemetryMessage:
    """Parse a MAVLink ``GPS_RAW_INT`` message.

    Converts lat/lon (×1e-7 deg) and alt (mm) into ENU position if an
    *enu_origin* is provided, otherwise stores raw WGS84.

    Args:
        msg: Dictionary with MAVLink fields (``lat``, ``lon``, ``alt``,
             ``eph``, ``epv``, ``fix_type``, ``time_usec``).
        node_id: Vehicle identifier.
        enu_origin: Optional ENU reference for coordinate conversion.
    """
    lat_deg = msg.get("lat", 0) * 1e-7
    lon_deg = msg.get("lon", 0) * 1e-7
    alt_m = msg.get("alt", 0) * 1e-3  # mm → m
    timestamp_s = msg.get("time_usec", 0) * 1e-6

    fields: dict[str, Any] = {
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "alt_m": alt_m,
        "eph_cm": msg.get("eph", 9999),
        "epv_cm": msg.get("epv", 9999),
        "fix_type": msg.get("fix_type", 0),
        "satellites_visible": msg.get("satellites_visible", 0),
    }

    if enu_origin is not None:
        enu = wgs84_to_enu(lat_deg, lon_deg, alt_m, enu_origin)
        fields["enu_m"] = [float(enu[0]), float(enu[1]), float(enu[2])]

    return TelemetryMessage(
        msg_type="GPS_RAW_INT",
        node_id=node_id,
        timestamp_s=timestamp_s,
        fields=fields,
    )


def parse_attitude(
    msg: dict[str, Any],
    node_id: str,
) -> TelemetryMessage:
    """Parse a MAVLink ``ATTITUDE`` message.

    Args:
        msg: Dictionary with ``roll``, ``pitch``, ``yaw`` (radians),
             ``rollspeed``, ``pitchspeed``, ``yawspeed`` (rad/s),
             ``time_boot_ms``.
    """
    return TelemetryMessage(
        msg_type="ATTITUDE",
        node_id=node_id,
        timestamp_s=msg.get("time_boot_ms", 0) * 1e-3,
        fields={
            "roll_rad": msg.get("roll", 0.0),
            "pitch_rad": msg.get("pitch", 0.0),
            "yaw_rad": msg.get("yaw", 0.0),
            "rollspeed_rad_s": msg.get("rollspeed", 0.0),
            "pitchspeed_rad_s": msg.get("pitchspeed", 0.0),
            "yawspeed_rad_s": msg.get("yawspeed", 0.0),
        },
    )


def parse_battery_status(
    msg: dict[str, Any],
    node_id: str,
) -> TelemetryMessage:
    """Parse a MAVLink ``BATTERY_STATUS`` message.

    Args:
        msg: Dictionary with ``voltages`` (list, mV), ``current_battery``
             (cA), ``battery_remaining`` (%), ``temperature`` (cdegC).
    """
    voltages_mv = msg.get("voltages", [])
    # MAVLink packs up to 10 cells; filter sentinels (0xFFFF = 65535)
    valid_voltages = [v for v in voltages_mv if 0 < v < 65535]
    total_voltage_v = sum(valid_voltages) * 1e-3 if valid_voltages else 0.0
    current_a = msg.get("current_battery", 0) * 1e-2  # cA → A

    return TelemetryMessage(
        msg_type="BATTERY_STATUS",
        node_id=node_id,
        timestamp_s=0.0,  # no timestamp in BATTERY_STATUS
        fields={
            "total_voltage_v": total_voltage_v,
            "current_a": current_a,
            "remaining_pct": msg.get("battery_remaining", -1),
            "temperature_c": msg.get("temperature", 0) * 0.01,
            "cell_count": len(valid_voltages),
        },
    )


def parse_heartbeat(
    msg: dict[str, Any],
    node_id: str,
) -> TelemetryMessage:
    """Parse a MAVLink ``HEARTBEAT`` message.

    Args:
        msg: Dictionary with ``type``, ``autopilot``, ``base_mode``,
             ``custom_mode``, ``system_status``.
    """
    return TelemetryMessage(
        msg_type="HEARTBEAT",
        node_id=node_id,
        timestamp_s=0.0,
        fields={
            "mav_type": msg.get("type", 0),
            "autopilot": msg.get("autopilot", 0),
            "base_mode": msg.get("base_mode", 0),
            "custom_mode": msg.get("custom_mode", 0),
            "system_status": msg.get("system_status", 0),
            "armed": bool(msg.get("base_mode", 0) & 128),
        },
    )


# ---------------------------------------------------------------------------
# Aggregator: combine multiple telemetry messages into NodeState
# ---------------------------------------------------------------------------


@dataclass
class _VehicleState:
    """Internal mutable state for a single vehicle."""

    node_id: str
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    attitude_rad: np.ndarray | None = None  # [roll, pitch, yaw]
    battery_pct: float = -1.0
    last_gps_timestamp_s: float = 0.0
    last_attitude_timestamp_s: float = 0.0
    is_armed: bool = False


class TelemetryAggregator:
    """Fuses successive telemetry messages into :class:`NodeState` snapshots.

    Maintains per-vehicle state and produces a ``NodeState`` whenever enough
    data has arrived (at minimum a GPS fix).

    Args:
        enu_origin: Optional ENU reference for WGS84 → local conversion.
    """

    def __init__(self, enu_origin: ENUOrigin | None = None) -> None:
        self._enu_origin = enu_origin
        self._vehicles: dict[str, _VehicleState] = {}

    def update(self, msg: TelemetryMessage) -> NodeState | None:
        """Ingest a single telemetry message and return an updated
        ``NodeState`` if sufficient data is available.
        """
        state = self._vehicles.get(msg.node_id)
        if state is None:
            state = _VehicleState(node_id=msg.node_id)
            self._vehicles[msg.node_id] = state

        if msg.msg_type == "GPS_RAW_INT":
            enu = msg.fields.get("enu_m")
            if enu is not None:
                prev = state.position
                state.position = np.array(enu, dtype=float)
                dt = msg.timestamp_s - state.last_gps_timestamp_s
                if prev is not None and dt > 0:
                    state.velocity = (state.position - prev) / dt
                elif state.velocity is None:
                    state.velocity = np.zeros(3)
                state.last_gps_timestamp_s = msg.timestamp_s
            else:
                lat = msg.fields.get("lat_deg", 0.0)
                lon = msg.fields.get("lon_deg", 0.0)
                alt = msg.fields.get("alt_m", 0.0)
                if self._enu_origin is not None:
                    enu_vec = wgs84_to_enu(lat, lon, alt, self._enu_origin)
                    state.position = np.array(enu_vec, dtype=float)
                state.last_gps_timestamp_s = msg.timestamp_s
                if state.velocity is None:
                    state.velocity = np.zeros(3)

        elif msg.msg_type == "ATTITUDE":
            state.attitude_rad = np.array(
                [
                    msg.fields.get("roll_rad", 0.0),
                    msg.fields.get("pitch_rad", 0.0),
                    msg.fields.get("yaw_rad", 0.0),
                ]
            )
            state.last_attitude_timestamp_s = msg.timestamp_s

        elif msg.msg_type == "BATTERY_STATUS":
            pct = msg.fields.get("remaining_pct", -1)
            if pct >= 0:
                state.battery_pct = float(pct)

        elif msg.msg_type == "HEARTBEAT":
            state.is_armed = msg.fields.get("armed", False)

        # Produce NodeState only if we have a position
        if state.position is None:
            return None

        vel = state.velocity if state.velocity is not None else np.zeros(3)
        return NodeState(
            node_id=state.node_id,
            position=vec3(state.position[0], state.position[1], state.position[2]),
            velocity=vec3(vel[0], vel[1], vel[2]),
            is_mobile=True,
            timestamp_s=state.last_gps_timestamp_s,
            health=max(state.battery_pct / 100.0, 0.0) if state.battery_pct >= 0 else 1.0,
        )

    def all_node_states(self) -> list[NodeState]:
        """Return the latest ``NodeState`` for every tracked vehicle."""
        result: list[NodeState] = []
        for state in self._vehicles.values():
            if state.position is not None:
                vel = state.velocity if state.velocity is not None else np.zeros(3)
                result.append(
                    NodeState(
                        node_id=state.node_id,
                        position=vec3(state.position[0], state.position[1], state.position[2]),
                        velocity=vec3(vel[0], vel[1], vel[2]),
                        is_mobile=True,
                        timestamp_s=state.last_gps_timestamp_s,
                        health=max(state.battery_pct / 100.0, 0.0)
                        if state.battery_pct >= 0
                        else 1.0,
                    )
                )
        return result

    def clear(self) -> None:
        self._vehicles.clear()


# ---------------------------------------------------------------------------
# JSON-lines replay adapter
# ---------------------------------------------------------------------------


class JSONTelemetryReplay:
    """Replays a JSON-lines file of telemetry messages.

    Each line should be a JSON object with at least ``msg_type``,
    ``node_id``, and the relevant fields for that message type.

    Example line::

        {"msg_type": "GPS_RAW_INT", "node_id": "drone-1",
         "lat": 474000000, "lon": 85000000, "alt": 50000}
    """

    def __init__(
        self,
        path: str,
        enu_origin: ENUOrigin | None = None,
    ) -> None:
        self._path = path
        self._aggregator = TelemetryAggregator(enu_origin=enu_origin)
        self._enu_origin = enu_origin

    def replay(self) -> list[NodeState]:
        """Read all lines and return final node states."""
        with open(self._path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                msg_dict = json.loads(line)
                msg_type = msg_dict.pop("msg_type", "")
                node_id = msg_dict.pop("node_id", "unknown")

                if msg_type == "GPS_RAW_INT":
                    tmsg = parse_gps_raw_int(msg_dict, node_id, self._enu_origin)
                elif msg_type == "ATTITUDE":
                    tmsg = parse_attitude(msg_dict, node_id)
                elif msg_type == "BATTERY_STATUS":
                    tmsg = parse_battery_status(msg_dict, node_id)
                elif msg_type == "HEARTBEAT":
                    tmsg = parse_heartbeat(msg_dict, node_id)
                else:
                    logger.debug("skipping unknown message type %s", msg_type)
                    continue

                self._aggregator.update(tmsg)

        return self._aggregator.all_node_states()
