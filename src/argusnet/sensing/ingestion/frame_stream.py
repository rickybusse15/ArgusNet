"""Live sensor ingestion adapters for MQTT and ROS 2."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from argusnet.core.frames import ENUOrigin, wgs84_to_enu
from argusnet.core.types import BearingObservation, NodeState, TruthState, vec3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

OnFrameCallback = Callable[
    [float, List[NodeState], List[BearingObservation], List[TruthState]],
    None,
]


class IngestionAdapter(Protocol):
    """Interface for live observation sources."""

    def start(self, on_frame: OnFrameCallback) -> None:
        """Begin producing observation frames."""
        ...

    def stop(self) -> None:
        """Stop the adapter and release resources."""
        ...


# ---------------------------------------------------------------------------
# MQTT message parsing
# ---------------------------------------------------------------------------

_REQUIRED_OBSERVATION_KEYS = frozenset(
    {"node_id", "azimuth_deg", "elevation_deg", "bearing_std_rad", "timestamp_unix_s"}
)


def parse_mqtt_observation(
    payload: Dict[str, Any],
    enu_origin: ENUOrigin,
) -> BearingObservation:
    """Convert a single MQTT JSON payload into a ``BearingObservation``.

    Expected fields:
        node_id           (str)   — sensor identifier
        target_id         (str, optional) — target label; defaults to "unknown"
        lat_deg           (float) — observer WGS84 latitude
        lon_deg           (float) — observer WGS84 longitude
        alt_m             (float) — observer altitude above WGS84 ellipsoid
        azimuth_deg       (float) — bearing azimuth from north, clockwise
        elevation_deg     (float) — bearing elevation above horizon
        bearing_std_rad   (float) — bearing noise standard deviation
        confidence        (float, optional) — observation confidence [0, 1]; defaults to 1.0
        timestamp_unix_s  (float) — UNIX epoch timestamp
    """
    missing = _REQUIRED_OBSERVATION_KEYS - payload.keys()
    if missing:
        raise ValueError(f"MQTT payload missing required keys: {sorted(missing)}")

    # Observer position in local ENU
    lat = float(payload.get("lat_deg", 0.0))
    lon = float(payload.get("lon_deg", 0.0))
    alt = float(payload.get("alt_m", 0.0))
    origin_enu = wgs84_to_enu(lat, lon, alt, enu_origin)

    # Convert azimuth/elevation to a unit direction vector in ENU
    az_rad = math.radians(float(payload["azimuth_deg"]))
    el_rad = math.radians(float(payload["elevation_deg"]))
    cos_el = math.cos(el_rad)
    # Azimuth: 0=North (positive Y), 90=East (positive X)
    dx = cos_el * math.sin(az_rad)
    dy = cos_el * math.cos(az_rad)
    dz = math.sin(el_rad)
    direction = vec3(dx, dy, dz)

    return BearingObservation(
        node_id=str(payload["node_id"]),
        target_id=str(payload.get("target_id", "unknown")),
        origin=origin_enu,
        direction=direction,
        bearing_std_rad=float(payload["bearing_std_rad"]),
        timestamp_s=float(payload["timestamp_unix_s"]),
        confidence=float(payload.get("confidence", 1.0)),
    )


def parse_mqtt_node_state(
    payload: Dict[str, Any],
    enu_origin: ENUOrigin,
) -> NodeState:
    """Convert an MQTT JSON payload into a ``NodeState``.

    Expected fields:
        node_id           (str)
        lat_deg, lon_deg, alt_m  (float) — WGS84 position
        vx_ms, vy_ms, vz_ms     (float, optional) — velocity in ENU, default 0
        is_mobile                (bool, optional) — default True
        timestamp_unix_s         (float)
        health                   (float, optional) — default 1.0
    """
    pos = wgs84_to_enu(
        float(payload["lat_deg"]),
        float(payload["lon_deg"]),
        float(payload["alt_m"]),
        enu_origin,
    )
    vel = vec3(
        float(payload.get("vx_ms", 0.0)),
        float(payload.get("vy_ms", 0.0)),
        float(payload.get("vz_ms", 0.0)),
    )
    return NodeState(
        node_id=str(payload["node_id"]),
        position=pos,
        velocity=vel,
        is_mobile=bool(payload.get("is_mobile", True)),
        timestamp_s=float(payload["timestamp_unix_s"]),
        health=float(payload.get("health", 1.0)),
    )


# ---------------------------------------------------------------------------
# MQTT Ingestion Adapter
# ---------------------------------------------------------------------------

@dataclass
class MQTTIngestionAdapter:
    """Subscribes to MQTT topics and converts payloads into observations.

    Requires the optional ``paho-mqtt`` dependency::

        pip install paho-mqtt

    Topics:
        * ``observation_topic`` — publishes ``BearingObservation`` JSON payloads
        * ``node_topic`` — publishes ``NodeState`` JSON payloads (optional)
    """

    broker: str
    port: int = 1883
    observation_topic: str = "smart_tracker/observations"
    node_topic: str = "smart_tracker/nodes"
    enu_origin: ENUOrigin = field(default_factory=lambda: ENUOrigin(0.0, 0.0, 0.0))

    _client: Any = field(default=None, init=False, repr=False)
    _on_frame: Optional[OnFrameCallback] = field(default=None, init=False, repr=False)
    _pending_observations: List[BearingObservation] = field(
        default_factory=list, init=False, repr=False
    )
    _pending_nodes: Dict[str, NodeState] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, on_frame: OnFrameCallback) -> None:
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise ImportError(
                "MQTT ingestion requires 'paho-mqtt'. Install with: "
                "pip install paho-mqtt"
            ) from exc

        self._on_frame = on_frame
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.connect(self.broker, self.port, keepalive=60)
        self._client.loop_start()

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        client.subscribe(self.observation_topic)
        client.subscribe(self.node_topic)
        logger.info("MQTT connected, subscribed to %s and %s", self.observation_topic, self.node_topic)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Failed to decode MQTT payload on %s: %s", msg.topic, exc)
            return

        try:
            if msg.topic == self.observation_topic:
                obs = parse_mqtt_observation(payload, self.enu_origin)
                with self._lock:
                    self._pending_observations.append(obs)
            elif msg.topic == self.node_topic:
                node = parse_mqtt_node_state(payload, self.enu_origin)
                with self._lock:
                    self._pending_nodes[node.node_id] = node
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to parse MQTT payload on %s: %s", msg.topic, exc)

    def flush_pending(self) -> tuple[List[NodeState], List[BearingObservation]]:
        """Remove and return buffered observations and node states."""
        with self._lock:
            nodes = list(self._pending_nodes.values())
            observations = list(self._pending_observations)
            self._pending_nodes.clear()
            self._pending_observations.clear()
        return nodes, observations


# ---------------------------------------------------------------------------
# File Replay Ingestion Adapter
# ---------------------------------------------------------------------------

def _vec3_from_list(seq: Any, fallback: tuple = (0.0, 0.0, 0.0)) -> "Vector3":
    """Convert a JSON-serialised ``[x, y, z]`` list into a ``vec3``.

    ``to_jsonable`` serialises ``np.ndarray`` as plain Python lists via
    ``.tolist()``, so replay vectors arrive as ``[x, y, z]``, not dicts.
    """
    try:
        return vec3(float(seq[0]), float(seq[1]), float(seq[2]))
    except (TypeError, IndexError, KeyError):
        return vec3(*fallback)


@dataclass
class FileReplayIngestionAdapter:
    """Replays a saved ``replay.json`` through the ingestion pipeline.

    Uses a **buffer/pull model** (same as :class:`MQTTIngestionAdapter`) so it
    is drop-in compatible with :class:`LiveIngestionRunner`.  A background
    thread reads frames from the file and pushes observations into an internal
    buffer; :meth:`flush_pending` drains that buffer on the caller's schedule.

    Args:
        replay_path: Path to the ``replay.json`` produced by ``smart-tracker sim``.
        speed: Playback rate multiplier (default 1.0 = real-time, 0 = as fast
               as possible).
        loop: If ``True``, restart playback when the file is exhausted.
    """

    replay_path: str
    speed: float = 1.0
    loop: bool = False

    _thread: Any = field(default=None, init=False, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _pending_observations: List[BearingObservation] = field(
        default_factory=list, init=False, repr=False
    )
    _pending_nodes: Dict[str, NodeState] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, on_frame: OnFrameCallback) -> None:
        """Start the background replay thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="FileReplayAdapter"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the replay thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def flush_pending(self) -> tuple[List[NodeState], List[BearingObservation]]:
        """Remove and return buffered node states and observations."""
        with self._lock:
            nodes = list(self._pending_nodes.values())
            observations = list(self._pending_observations)
            self._pending_nodes.clear()
            self._pending_observations.clear()
        return nodes, observations

    def _run(self) -> None:
        import json as _json

        with open(self.replay_path, "r") as fh:
            data = _json.load(fh)

        frames = data.get("frames", [])
        if not frames:
            logger.warning("FileReplayIngestionAdapter: no frames in %s", self.replay_path)
            return

        while not self._stop_event.is_set():
            prev_ts: Optional[float] = None
            for frame_data in frames:
                if self._stop_event.is_set():
                    return

                timestamp_s: float = float(frame_data.get("timestamp_s", 0.0))

                # Parse node states — vectors are [x, y, z] lists (from to_jsonable)
                for n in frame_data.get("nodes", []):
                    node = NodeState(
                        node_id=str(n["node_id"]),
                        position=_vec3_from_list(n.get("position", [0.0, 0.0, 0.0])),
                        velocity=_vec3_from_list(n.get("velocity", [0.0, 0.0, 0.0])),
                        is_mobile=bool(n.get("is_mobile", False)),
                        timestamp_s=float(n.get("timestamp_s", timestamp_s)),
                        health=float(n.get("health", 1.0)),
                    )
                    with self._lock:
                        self._pending_nodes[node.node_id] = node

                # Parse bearing observations
                obs_batch: List[BearingObservation] = []
                for o in frame_data.get("observations", []):
                    obs_batch.append(BearingObservation(
                        node_id=str(o["node_id"]),
                        target_id=str(o.get("target_id", "unknown")),
                        origin=_vec3_from_list(o.get("origin", [0.0, 0.0, 0.0])),
                        direction=_vec3_from_list(o.get("direction", [0.0, 0.0, 1.0])),
                        bearing_std_rad=float(o.get("bearing_std_rad", 0.01)),
                        timestamp_s=float(o.get("timestamp_s", timestamp_s)),
                        confidence=float(o.get("confidence", 1.0)),
                    ))
                if obs_batch:
                    with self._lock:
                        self._pending_observations.extend(obs_batch)

                logger.debug(
                    "FileReplayIngestionAdapter: buffered frame t=%.3f (%d obs)",
                    timestamp_s, len(obs_batch),
                )

                # Honour inter-frame timing
                if prev_ts is not None and self.speed > 0.0:
                    delay = (timestamp_s - prev_ts) / self.speed
                    if delay > 0.0:
                        self._stop_event.wait(delay)
                prev_ts = timestamp_s

            if not self.loop:
                break
            logger.info("FileReplayIngestionAdapter: looping replay of %s", self.replay_path)


# ---------------------------------------------------------------------------
# Live Ingestion Runner
# ---------------------------------------------------------------------------

class LiveIngestionRunner:
    """Orchestrates an adapter + a TrackingService for real-time ingestion.

    Periodically flushes buffered observations from the adapter and sends them
    to the tracking daemon via ``TrackingService.ingest_frame()``.

    Args:
        adapter: A started :class:`MQTTIngestionAdapter` (or compatible).
        service: An already-connected :class:`~smart_tracker.service.TrackingService`.
        frame_interval_s: How often to flush and ingest a frame (seconds).
        replay_frames: If provided, frames are appended here for replay export.
    """

    def __init__(
        self,
        adapter: MQTTIngestionAdapter,
        service: Any,  # TrackingService — avoid circular import
        frame_interval_s: float = 0.25,
        replay_frames: Optional[list] = None,
    ) -> None:
        self.adapter = adapter
        self.service = service
        self.frame_interval_s = frame_interval_s
        self.replay_frames = replay_frames
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Block and ingest frames until :meth:`stop` is called."""
        logger.info(
            "LiveIngestionRunner started (frame_interval=%.3fs)", self.frame_interval_s
        )
        while not self._stop_event.is_set():
            start = time.monotonic()
            nodes, observations = self.adapter.flush_pending()
            if observations:
                timestamp_s = observations[0].timestamp_s
                frame = self.service.ingest_frame(
                    timestamp_s=timestamp_s,
                    node_states=nodes or None,
                    observations=observations,
                )
                if self.replay_frames is not None:
                    self.replay_frames.append(frame)
                logger.debug(
                    "Ingested frame t=%.3f with %d observations, %d tracks",
                    timestamp_s,
                    len(observations),
                    len(frame.tracks),
                )
            elapsed = time.monotonic() - start
            remaining = self.frame_interval_s - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

    def stop(self) -> None:
        self._stop_event.set()
