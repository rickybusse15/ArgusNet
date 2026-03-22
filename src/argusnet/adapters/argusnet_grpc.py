from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import warnings

import grpc
import numpy as np

from smarttracker.v1 import tracker_pb2, tracker_pb2_grpc

from argusnet.core.types import (
    BearingObservation,
    HealthReport,
    NodeHealthMetrics,
    NodeState,
    ObservationRejection,
    PlatformFrame,
    PlatformMetrics,
    TrackState,
    TruthState,
)


REJECT_UNKNOWN_NODE = "unknown_node"
REJECT_INVALID_TARGET = "invalid_target_id"
REJECT_INVALID_DIRECTION = "invalid_direction"
REJECT_LOW_CONFIDENCE = "low_confidence"
REJECT_INVALID_BEARING_STD = "invalid_bearing_std"
REJECT_EXCESS_BEARING_STD = "bearing_noise_too_high"
REJECT_TIMESTAMP_SKEW = "timestamp_skew"
REJECT_DUPLICATE_NODE = "duplicate_node_observation"
REJECT_INSUFFICIENT_CLUSTER = "insufficient_cluster_observations"
REJECT_WEAK_GEOMETRY = "weak_intersection_geometry"
REJECT_FUSION_FAILURE = "fusion_failure"

_GRPC_CHANNEL_OPTIONS = (
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
)


@dataclass(frozen=True)
class TrackerConfig:
    min_observations: int = 2
    max_stale_steps: int = 8
    retain_history: bool = False
    min_confidence: float = 0.15
    max_bearing_std_rad: float = 0.08
    max_timestamp_skew_s: float = 1.5
    min_intersection_angle_deg: float = 2.5
    data_association_mode: str = "labeled"
    cv_process_accel_std: float = 3.0
    ct_process_accel_std: float = 8.0
    ct_turn_rate_std: float = 0.1
    innovation_window: int = 5
    innovation_scale_factor: float = 1.5
    innovation_max_scale: float = 4.0
    adaptive_measurement_noise: bool = False
    chi_squared_gate_threshold: float = 16.0
    cluster_distance_threshold_m: float = 200.0
    near_parallel_rejection_angle_deg: float = 2.5
    confirmation_m: int = 3
    confirmation_n: int = 5
    max_coast_frames: int = 10
    max_coast_seconds: float = 5.0
    min_quality_score: float = 0.1

    def __post_init__(self) -> None:
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2.")
        if self.max_stale_steps < 0:
            raise ValueError("max_stale_steps must be >= 0.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within [0.0, 1.0].")
        if not np.isfinite(self.max_bearing_std_rad) or self.max_bearing_std_rad <= 0.0:
            raise ValueError("max_bearing_std_rad must be finite and > 0.")
        if not np.isfinite(self.max_timestamp_skew_s) or self.max_timestamp_skew_s < 0.0:
            raise ValueError("max_timestamp_skew_s must be finite and >= 0.")
        if not np.isfinite(self.min_intersection_angle_deg) or self.min_intersection_angle_deg <= 0.0:
            raise ValueError("min_intersection_angle_deg must be finite and > 0.")
        if self.data_association_mode not in {"labeled", "gnn", "jpda"}:
            raise ValueError("data_association_mode must be one of labeled, gnn, jpda.")
        for field_name, value in (
            ("cv_process_accel_std", self.cv_process_accel_std),
            ("ct_process_accel_std", self.ct_process_accel_std),
            ("ct_turn_rate_std", self.ct_turn_rate_std),
            ("innovation_scale_factor", self.innovation_scale_factor),
            ("innovation_max_scale", self.innovation_max_scale),
            ("chi_squared_gate_threshold", self.chi_squared_gate_threshold),
            ("cluster_distance_threshold_m", self.cluster_distance_threshold_m),
            ("near_parallel_rejection_angle_deg", self.near_parallel_rejection_angle_deg),
            ("max_coast_seconds", self.max_coast_seconds),
            ("min_quality_score", self.min_quality_score),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite.")
        if self.cv_process_accel_std <= 0.0 or self.ct_process_accel_std <= 0.0:
            raise ValueError("CV/CT process noise must be > 0.")
        if self.ct_turn_rate_std <= 0.0:
            raise ValueError("ct_turn_rate_std must be > 0.")
        if self.innovation_window < 1:
            raise ValueError("innovation_window must be at least 1.")
        if self.innovation_scale_factor < 1.0 or self.innovation_max_scale < 1.0:
            raise ValueError("innovation adaptation scale factors must be >= 1.0.")
        if self.chi_squared_gate_threshold <= 0.0:
            raise ValueError("chi_squared_gate_threshold must be > 0.")
        if self.cluster_distance_threshold_m <= 0.0:
            raise ValueError("cluster_distance_threshold_m must be > 0.")
        if self.near_parallel_rejection_angle_deg <= 0.0:
            raise ValueError("near_parallel_rejection_angle_deg must be > 0.")
        if self.confirmation_m < 1 or self.confirmation_n < self.confirmation_m:
            raise ValueError("confirmation_n must be >= confirmation_m >= 1.")
        if self.max_coast_frames < 0:
            raise ValueError("max_coast_frames must be >= 0.")
        if self.max_coast_seconds < 0.0:
            raise ValueError("max_coast_seconds must be >= 0.")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be within [0.0, 1.0].")


def _vector_to_proto(vector: np.ndarray) -> tracker_pb2.Vector3:
    value = np.asarray(vector, dtype=float).reshape(3)
    return tracker_pb2.Vector3(x_m=float(value[0]), y_m=float(value[1]), z_m=float(value[2]))


def _vector_from_proto(vector: tracker_pb2.Vector3) -> np.ndarray:
    return np.array([vector.x_m, vector.y_m, vector.z_m], dtype=float)


def _optional_vector_from_proto(vector: Optional[tracker_pb2.Vector3]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    return _vector_from_proto(vector)


def _tracker_config_to_proto(config: TrackerConfig) -> tracker_pb2.TrackerConfig:
    return tracker_pb2.TrackerConfig(
        min_observations=int(config.min_observations),
        max_stale_steps=int(config.max_stale_steps),
        retain_history=bool(config.retain_history),
        min_confidence=float(config.min_confidence),
        max_bearing_std_rad=float(config.max_bearing_std_rad),
        max_timestamp_skew_s=float(config.max_timestamp_skew_s),
        min_intersection_angle_deg=float(config.min_intersection_angle_deg),
        data_association_mode=str(config.data_association_mode),
        cv_process_accel_std=float(config.cv_process_accel_std),
        ct_process_accel_std=float(config.ct_process_accel_std),
        ct_turn_rate_std=float(config.ct_turn_rate_std),
        innovation_window=int(config.innovation_window),
        innovation_scale_factor=float(config.innovation_scale_factor),
        innovation_max_scale=float(config.innovation_max_scale),
        adaptive_measurement_noise=bool(config.adaptive_measurement_noise),
        chi_squared_gate_threshold=float(config.chi_squared_gate_threshold),
        cluster_distance_threshold_m=float(config.cluster_distance_threshold_m),
        near_parallel_rejection_angle_deg=float(config.near_parallel_rejection_angle_deg),
        confirmation_m=int(config.confirmation_m),
        confirmation_n=int(config.confirmation_n),
        max_coast_frames=int(config.max_coast_frames),
        max_coast_seconds=float(config.max_coast_seconds),
        min_quality_score=float(config.min_quality_score),
    )


def _tracker_config_from_proto(message: tracker_pb2.TrackerConfig) -> TrackerConfig:
    return TrackerConfig(
        min_observations=int(message.min_observations),
        max_stale_steps=int(message.max_stale_steps),
        retain_history=bool(message.retain_history),
        min_confidence=float(message.min_confidence),
        max_bearing_std_rad=float(message.max_bearing_std_rad),
        max_timestamp_skew_s=float(message.max_timestamp_skew_s),
        min_intersection_angle_deg=float(message.min_intersection_angle_deg),
        data_association_mode=str(message.data_association_mode or "labeled"),
        cv_process_accel_std=float(message.cv_process_accel_std or 3.0),
        ct_process_accel_std=float(message.ct_process_accel_std or 8.0),
        ct_turn_rate_std=float(message.ct_turn_rate_std or 0.1),
        innovation_window=int(message.innovation_window or 5),
        innovation_scale_factor=float(message.innovation_scale_factor or 1.5),
        innovation_max_scale=float(message.innovation_max_scale or 4.0),
        adaptive_measurement_noise=bool(message.adaptive_measurement_noise),
        chi_squared_gate_threshold=float(message.chi_squared_gate_threshold or 16.0),
        cluster_distance_threshold_m=float(message.cluster_distance_threshold_m or 200.0),
        near_parallel_rejection_angle_deg=float(message.near_parallel_rejection_angle_deg or 2.5),
        confirmation_m=int(message.confirmation_m or 3),
        confirmation_n=int(message.confirmation_n or 5),
        max_coast_frames=int(message.max_coast_frames or 10),
        max_coast_seconds=float(message.max_coast_seconds or 5.0),
        min_quality_score=float(message.min_quality_score or 0.1),
    )


def _node_to_proto(node: NodeState) -> tracker_pb2.NodeState:
    return tracker_pb2.NodeState(
        node_id=node.node_id,
        position=_vector_to_proto(node.position),
        velocity=_vector_to_proto(node.velocity),
        is_mobile=bool(node.is_mobile),
        timestamp_s=float(node.timestamp_s),
        health=float(node.health),
    )


def _node_from_proto(node: tracker_pb2.NodeState) -> NodeState:
    return NodeState(
        node_id=node.node_id,
        position=_vector_from_proto(node.position),
        velocity=_vector_from_proto(node.velocity),
        is_mobile=bool(node.is_mobile),
        timestamp_s=float(node.timestamp_s),
        health=float(node.health),
    )


def _observation_to_proto(observation: BearingObservation) -> tracker_pb2.BearingObservation:
    return tracker_pb2.BearingObservation(
        node_id=observation.node_id,
        target_id=observation.target_id,
        origin=_vector_to_proto(observation.origin),
        direction=_vector_to_proto(observation.direction),
        bearing_std_rad=float(observation.bearing_std_rad),
        timestamp_s=float(observation.timestamp_s),
        confidence=float(observation.confidence),
    )


def _observation_from_proto(observation: tracker_pb2.BearingObservation) -> BearingObservation:
    return BearingObservation(
        node_id=observation.node_id,
        target_id=observation.target_id,
        origin=_vector_from_proto(observation.origin),
        direction=_vector_from_proto(observation.direction),
        bearing_std_rad=float(observation.bearing_std_rad),
        timestamp_s=float(observation.timestamp_s),
        confidence=float(observation.confidence),
    )


def _rejection_from_proto(rejection: tracker_pb2.ObservationRejection) -> ObservationRejection:
    return ObservationRejection(
        node_id=rejection.node_id,
        target_id=rejection.target_id,
        timestamp_s=float(rejection.timestamp_s),
        reason=rejection.reason,
        detail=rejection.detail,
        origin=_optional_vector_from_proto(rejection.origin if rejection.HasField("origin") else None),
        attempted_point=_optional_vector_from_proto(
            rejection.attempted_point if rejection.HasField("attempted_point") else None
        ),
        closest_point=_optional_vector_from_proto(
            rejection.closest_point if rejection.HasField("closest_point") else None
        ),
        blocker_type=rejection.blocker_type,
        first_hit_range_m=_optional_double(rejection, "first_hit_range_m"),
    )


def _truth_to_proto(truth: TruthState) -> tracker_pb2.TruthState:
    return tracker_pb2.TruthState(
        target_id=truth.target_id,
        position=_vector_to_proto(truth.position),
        velocity=_vector_to_proto(truth.velocity),
        timestamp_s=float(truth.timestamp_s),
    )


def _truth_from_proto(truth: tracker_pb2.TruthState) -> TruthState:
    return TruthState(
        target_id=truth.target_id,
        position=_vector_from_proto(truth.position),
        velocity=_vector_from_proto(truth.velocity),
        timestamp_s=float(truth.timestamp_s),
    )


def _track_from_proto(track: tracker_pb2.TrackState) -> TrackState:
    covariance = np.asarray(track.covariance_row_major, dtype=float).reshape(6, 6)
    return TrackState(
        track_id=track.track_id,
        timestamp_s=float(track.timestamp_s),
        position=_vector_from_proto(track.position),
        velocity=_vector_from_proto(track.velocity),
        covariance=covariance,
        measurement_std_m=float(track.measurement_std_m),
        update_count=int(track.update_count),
        stale_steps=int(track.stale_steps),
        lifecycle_state=str(track.lifecycle_state) if track.HasField("lifecycle_state") else None,
        quality_score=_optional_double(track, "quality_score"),
    )


def _optional_double(message: object, field_name: str) -> Optional[float]:
    if hasattr(message, "HasField") and message.HasField(field_name):
        return float(getattr(message, field_name))
    return None


def _metrics_from_proto(metrics: tracker_pb2.PlatformMetrics) -> PlatformMetrics:
    return PlatformMetrics(
        mean_error_m=_optional_double(metrics, "mean_error_m"),
        max_error_m=_optional_double(metrics, "max_error_m"),
        active_track_count=int(metrics.active_track_count),
        observation_count=int(metrics.observation_count),
        accepted_observation_count=int(metrics.accepted_observation_count),
        rejected_observation_count=int(metrics.rejected_observation_count),
        mean_measurement_std_m=_optional_double(metrics, "mean_measurement_std_m"),
        track_errors_m={key: float(value) for key, value in metrics.track_errors_m.items()},
        rejection_counts={key: int(value) for key, value in metrics.rejection_counts.items()},
        accepted_observations_by_target={
            key: int(value) for key, value in metrics.accepted_observations_by_target.items()
        },
        rejected_observations_by_target={
            key: int(value) for key, value in metrics.rejected_observations_by_target.items()
        },
    )


def _frame_from_proto(frame: tracker_pb2.PlatformFrame) -> PlatformFrame:
    return PlatformFrame(
        timestamp_s=float(frame.timestamp_s),
        nodes=[_node_from_proto(node) for node in frame.nodes],
        observations=[_observation_from_proto(observation) for observation in frame.observations],
        rejected_observations=[
            _rejection_from_proto(rejection) for rejection in frame.rejected_observations
        ],
        tracks=[_track_from_proto(track) for track in frame.tracks],
        truths=[_truth_from_proto(truth) for truth in frame.truths],
        metrics=_metrics_from_proto(frame.metrics),
        generation_rejections=[
            _rejection_from_proto(rejection) for rejection in frame.generation_rejections
        ],
    )


def _cleanup_service_resources(
    channel: Optional[grpc.Channel],
    process: Optional[subprocess.Popen[str]],
    tempdir: Optional[tempfile.TemporaryDirectory[str]],
    log_handle: Optional[object],
) -> None:
    if channel is not None:
        try:
            channel.close()
        except Exception:
            pass
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
    if tempdir is not None:
        tempdir.cleanup()
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:
            pass


class TrackingService:
    def __init__(
        self,
        config: Optional[TrackerConfig] = None,
        *,
        retain_history: Optional[bool] = None,
        endpoint: Optional[str] = None,
        spawn_local: Optional[bool] = None,
        daemon_path: Optional[str] = None,
        startup_timeout_s: float = 20.0,
    ) -> None:
        self.config = config or TrackerConfig()
        self.retain_history = self.config.retain_history if retain_history is None else retain_history
        self.nodes: Dict[str, NodeState] = {}
        self.tracks: Dict[str, TrackState] = {}
        self.history: List[PlatformFrame] = []
        self._latest_frame: Optional[PlatformFrame] = None
        self._owned_process: Optional[subprocess.Popen[str]] = None
        self._owned_tempdir: Optional[tempfile.TemporaryDirectory[str]] = None
        self._owned_log_handle: Optional[object] = None

        should_spawn_local = endpoint is None if spawn_local is None else spawn_local
        if should_spawn_local:
            self.endpoint = endpoint or self._spawn_local_daemon(daemon_path)
        else:
            self.endpoint = endpoint
        if self.endpoint is None:
            raise ValueError("TrackingService requires an endpoint or spawn_local=True.")

        if self.endpoint and not self.endpoint.startswith("127.0.0.1") and not self.endpoint.startswith("localhost") and not self.endpoint.startswith("[::1]"):
            warnings.warn(
                f"Connecting to non-localhost endpoint {self.endpoint!r} without TLS. "
                "Data will be transmitted in plaintext.",
                stacklevel=2,
            )
        self._channel = grpc.insecure_channel(self.endpoint, options=_GRPC_CHANNEL_OPTIONS)
        self._stub = tracker_pb2_grpc.TrackerServiceStub(self._channel)
        self._finalizer = weakref.finalize(
            self,
            _cleanup_service_resources,
            self._channel,
            self._owned_process,
            self._owned_tempdir,
            self._owned_log_handle,
        )

        self._wait_for_ready(startup_timeout_s)
        remote_config = self._stub.GetConfig(tracker_pb2.GetConfigRequest(), timeout=startup_timeout_s)
        if remote_config.HasField("config"):
            self.remote_config = _tracker_config_from_proto(remote_config.config)
        else:
            self.remote_config = self.config

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _spawn_local_daemon(self, daemon_path: Optional[str]) -> str:
        endpoint = self._allocate_endpoint()
        host, port = endpoint.split(":")
        self._owned_tempdir = tempfile.TemporaryDirectory(prefix="smart-trackerd-")
        config_path = Path(self._owned_tempdir.name) / "tracker-config.yaml"
        log_path = Path(self._owned_tempdir.name) / "tracker.log"
        config_path.write_text(self._render_tracker_config_yaml(self.config), encoding="utf-8")

        command = self._resolve_daemon_command(daemon_path)
        self._owned_log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [*command, "serve", "--listen", f"{host}:{port}", "--config", str(config_path)],
            cwd=self._repo_root(),
            stdout=self._owned_log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._owned_process = process
        return endpoint

    @staticmethod
    def _allocate_endpoint() -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
            handle.bind(("127.0.0.1", 0))
            host, port = handle.getsockname()
        return f"{host}:{port}"

    @staticmethod
    def _render_tracker_config_yaml(config: TrackerConfig) -> str:
        return "\n".join(
            [
                f"min_observations: {config.min_observations}",
                f"max_stale_steps: {config.max_stale_steps}",
                f"retain_history: {str(config.retain_history).lower()}",
                f"min_confidence: {config.min_confidence}",
                f"max_bearing_std_rad: {config.max_bearing_std_rad}",
                f"max_timestamp_skew_s: {config.max_timestamp_skew_s}",
                f"min_intersection_angle_deg: {config.min_intersection_angle_deg}",
                f"data_association_mode: {config.data_association_mode}",
                f"cv_process_accel_std: {config.cv_process_accel_std}",
                f"ct_process_accel_std: {config.ct_process_accel_std}",
                f"ct_turn_rate_std: {config.ct_turn_rate_std}",
                f"innovation_window: {config.innovation_window}",
                f"innovation_scale_factor: {config.innovation_scale_factor}",
                f"innovation_max_scale: {config.innovation_max_scale}",
                f"adaptive_measurement_noise: {str(config.adaptive_measurement_noise).lower()}",
                f"chi_squared_gate_threshold: {config.chi_squared_gate_threshold}",
                f"cluster_distance_threshold_m: {config.cluster_distance_threshold_m}",
                f"near_parallel_rejection_angle_deg: {config.near_parallel_rejection_angle_deg}",
                f"confirmation_m: {config.confirmation_m}",
                f"confirmation_n: {config.confirmation_n}",
                f"max_coast_frames: {config.max_coast_frames}",
                f"max_coast_seconds: {config.max_coast_seconds}",
                f"min_quality_score: {config.min_quality_score}",
                "",
            ]
        )

    def _resolve_daemon_command(self, daemon_path: Optional[str]) -> list[str]:
        repo_root = self._repo_root()
        if daemon_path:
            return [daemon_path]

        env_daemon = shutil.which("smart-trackerd")
        if env_daemon:
            return [env_daemon]

        debug_binary = repo_root / "target" / "debug" / "smart-trackerd"
        if not debug_binary.exists():
            cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
            print("Building smart-trackerd (this may take a minute on first run)...")
            subprocess.run(
                [cargo, "build", "--manifest-path", str(repo_root / "Cargo.toml"), "-p", "tracker-server"],
                cwd=repo_root,
                check=True,
            )
        if debug_binary.exists():
            return [str(debug_binary)]

        cargo = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / "cargo")
        return [
            cargo,
            "run",
            "--manifest-path",
            str(repo_root / "Cargo.toml"),
            "-p",
            "tracker-server",
            "--bin",
            "smart-trackerd",
            "--",
        ]

    def _wait_for_ready(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        last_error: Optional[BaseException] = None
        while time.time() < deadline:
            if self._owned_process is not None and self._owned_process.poll() is not None:
                raise RuntimeError(
                    f"smart-trackerd exited early with code {self._owned_process.returncode}."
                )
            try:
                self._stub.Health(tracker_pb2.HealthRequest(), timeout=0.5)
                return
            except Exception as error:  # pragma: no cover - retry path depends on startup timing.
                last_error = error
                time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for smart-trackerd at {self.endpoint}: {last_error}")

    def close(self) -> None:
        if self._finalizer.alive:
            self._finalizer()

    def __enter__(self) -> "TrackingService":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup.
        try:
            self.close()
        except Exception:
            pass

    def ingest_frame(
        self,
        timestamp_s: float,
        node_states: Optional[Sequence[NodeState]] = None,
        observations: Iterable[BearingObservation] = (),
        truths: Optional[Sequence[TruthState]] = None,
    ) -> PlatformFrame:
        request = tracker_pb2.IngestFrameRequest(
            timestamp_s=float(timestamp_s),
            node_states=[_node_to_proto(node) for node in (node_states or ())],
            observations=[_observation_to_proto(observation) for observation in observations],
            truths=[_truth_to_proto(truth) for truth in (truths or ())],
        )
        response = self._stub.IngestFrame(request)
        if not response.HasField("frame"):
            raise RuntimeError("tracking daemon returned an empty frame response")

        frame = _frame_from_proto(response.frame)
        self._latest_frame = frame
        self.nodes = {node.node_id: node for node in frame.nodes}
        self.tracks = {track.track_id: track for track in frame.tracks}
        if self.retain_history:
            self.history.append(frame)
        return frame

    def track_stream(
        self,
        frames: Iterable[
            tuple[float, Optional[Sequence[NodeState]], Iterable[BearingObservation], Optional[Sequence[TruthState]]]
        ],
    ) -> Iterator[PlatformFrame]:
        """Stream frames to the daemon and yield one PlatformFrame per input frame.

        ``frames`` is an iterable of ``(timestamp_s, node_states, observations, truths)`` tuples.
        Uses the bidirectional-streaming ``TrackStream`` RPC for lower per-frame overhead than
        repeated ``IngestFrame`` calls.
        """

        def _request_iter() -> Iterator[tracker_pb2.IngestFrameRequest]:
            for timestamp_s, node_states, observations, truths in frames:
                yield tracker_pb2.IngestFrameRequest(
                    timestamp_s=float(timestamp_s),
                    node_states=[_node_to_proto(n) for n in (node_states or ())],
                    observations=[_observation_to_proto(o) for o in observations],
                    truths=[_truth_to_proto(t) for t in (truths or ())],
                )

        for response in self._stub.TrackStream(_request_iter()):
            if not response.HasField("frame"):
                raise RuntimeError("tracking daemon returned an empty frame response")
            frame = _frame_from_proto(response.frame)
            self._latest_frame = frame
            self.nodes = {node.node_id: node for node in frame.nodes}
            self.tracks = {track.track_id: track for track in frame.tracks}
            if self.retain_history:
                self.history.append(frame)
            yield frame

    def latest_frame(self) -> Optional[PlatformFrame]:
        if self._latest_frame is not None:
            return self._latest_frame
        response = self._stub.LatestFrame(tracker_pb2.LatestFrameRequest())
        if response.HasField("frame"):
            self._latest_frame = _frame_from_proto(response.frame)
            self.nodes = {node.node_id: node for node in self._latest_frame.nodes}
            self.tracks = {track.track_id: track for track in self._latest_frame.tracks}
        return self._latest_frame

    def reset(self) -> None:
        self._stub.Reset(tracker_pb2.ResetRequest())
        self.nodes.clear()
        self.tracks.clear()
        self.history.clear()
        self._latest_frame = None

    def health(self) -> HealthReport:
        response = self._stub.Health(tracker_pb2.HealthRequest(), timeout=5.0)
        node_health = [
            NodeHealthMetrics(
                node_id=nh.node_id,
                last_seen_s=float(nh.last_seen_s),
                observation_rate_hz=float(nh.observation_rate_hz),
                mean_latency_s=float(nh.mean_latency_s),
                accepted_count=int(nh.accepted_count),
                rejected_count=int(nh.rejected_count),
                health_score=float(nh.health_score),
            )
            for nh in response.node_health
        ]
        return HealthReport(
            status=response.status,
            started_at_utc=response.started_at_utc,
            processed_frame_count=int(response.processed_frame_count),
            node_health=node_health,
            mean_frame_rate_hz=float(response.mean_frame_rate_hz),
            mean_ingest_latency_s=float(response.mean_ingest_latency_s),
            active_node_count=int(response.active_node_count),
            stale_node_count=int(response.stale_node_count),
        )

# ArgusNet canonical aliases
WorldModelConfig = TrackerConfig
WorldModelService = TrackingService
