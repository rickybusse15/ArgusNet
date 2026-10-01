"""Direct fusion parity (P4.9c): the Rust tick's in-process fusion is the
daemon's fusion, message for message.

The reference fuses through ``argusnetd`` over gRPC; the Rust tick calls
``TrackingEngine::ingest_frame`` in process (P4.8c). This records every
``IngestFrameRequest`` a real reference run sends, and every
``IngestFrameResponse`` the daemon returns, then replays the requests through
``argusnet-rs fuse`` -- the same proto conversion and engine call, no
transport -- and requires every response to be the same message.
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from argusnet.adapters import argusnet_grpc
from argusnet.adapters.argusnet_grpc import TrackerConfig, TrackingService, tracker_pb2
from argusnet.core.types import (
    BearingObservation,
    NodeState,
    SafetyValidationResult,
    TargetMetadata,
    TruthState,
    vec3,
)
from argusnet.simulation.sim import (
    DEFAULT_SIMULATION_TRACKER_CONFIG,
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


def _argusnet_rs():
    from test_scenario_bundle import _argusnet_rs as locate

    return locate()


class _RecordingQueue:
    def __init__(self, inner, sink):
        self._inner = inner
        self._sink = sink

    def put(self, item):
        if isinstance(item, tracker_pb2.IngestFrameRequest):
            self._sink.append(item.SerializeToString())
        self._inner.put(item)

    def get(self, *args, **kwargs):
        return self._inner.get(*args, **kwargs)


@pytest.fixture
def recorded(monkeypatch):
    """Every tracker config, request and response that crosses an
    ``IngestStream`` while the test runs."""
    record = {"configs": [], "requests": [], "responses": []}
    original_init = argusnet_grpc.IngestStream.__init__

    def recording_init(self, service):
        original_init(self, service)
        record["configs"].append(
            argusnet_grpc._tracker_config_to_proto(service.config).SerializeToString()
        )
        self._requests = _RecordingQueue(self._requests, record["requests"])
        inner = self._responses

        def responses():
            for response in inner:
                record["responses"].append(response.SerializeToString())
                yield response

        self._responses = responses()

    monkeypatch.setattr(argusnet_grpc.IngestStream, "__init__", recording_init)
    return record


def _fuse_and_compare(tmp_path, record) -> list[tracker_pb2.IngestFrameResponse]:
    """Replays the recorded requests through ``argusnet-rs fuse`` and requires
    every response to be the daemon's message. Returns the daemon's."""
    binary = _argusnet_rs()
    assert len(record["configs"]) == 1
    assert len(record["requests"]) == len(record["responses"]) > 0
    capture = tmp_path / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "tracker": record["configs"][0].hex(),
                "requests": [r.hex() for r in record["requests"]],
            }
        )
    )
    output = tmp_path / "fused.json"
    completed = subprocess.run(
        [str(binary), "fuse", "--capture", str(capture), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    fused = json.loads(output.read_text())["responses"]
    assert len(fused) == len(record["responses"])

    daemon = []
    for index, (ours, theirs) in enumerate(zip(fused, record["responses"], strict=True)):
        ours_message = tracker_pb2.IngestFrameResponse.FromString(bytes.fromhex(ours))
        theirs_message = tracker_pb2.IngestFrameResponse.FromString(theirs)
        if ours_message != theirs_message:
            pytest.fail(
                f"frame {index}: the in-process engine answered differently\n"
                f"daemon: {theirs_message.frame}\nrust: {ours_message.frame}"
            )
        daemon.append(theirs_message)
    return daemon


def test_the_in_process_engine_answers_every_request_as_the_daemon_does(tmp_path, recorded):
    scenario = build_default_scenario(
        ScenarioOptions(
            map_preset="small",
            mission_mode="target_tracking",
            target_count=2,
            drone_count=3,
            ground_station_count=4,
        ),
        seed=17,
    )
    run_simulation(
        scenario,
        SimulationConfig(steps=60, dt_s=0.5, seed=17),
        tracker_config=DEFAULT_SIMULATION_TRACKER_CONFIG,
    )
    assert len(recorded["requests"]) == 60
    responses = _fuse_and_compare(tmp_path, recorded)
    tracked_frames = sum(bool(response.frame.tracks) for response in responses)
    assert tracked_frames > 20, "too few frames with tracks to mean anything"


def _bearing(node: NodeState, target_id: str, target, std: float) -> BearingObservation:
    direction = target - node.position
    return BearingObservation(
        node_id=node.node_id,
        target_id=target_id,
        origin=node.position,
        direction=direction / np.linalg.norm(direction),
        bearing_std_rad=std,
        timestamp_s=node.timestamp_s,
        confidence=1.0,
    )


def test_what_a_simulation_run_leaves_out_crosses_too(tmp_path, recorded):
    """A simulation run without blocking safety sends no safety events, and
    its tracker config differs from the engine default only where 60 ticks
    never reach. So these frames carry what that run does not: operator
    metadata, a safety event, and a tracker config that rejects one node's
    observations (0.05 rad, over this config's 0.02 but under the default)."""
    service = TrackingService(TrackerConfig(max_bearing_std_rad=0.02), retain_history=False)
    try:
        stream = service.open_ingest_stream()
        try:
            for step in range(8):
                t = 1.0 + 0.5 * step
                target = vec3(50.0 + 2.0 * step, 40.0, 20.0)
                nodes = [
                    NodeState("gs-a", vec3(0, 0, 0), vec3(0, 0, 0), False, t),
                    NodeState("gs-b", vec3(100, 0, 0), vec3(0, 0, 0), False, t),
                    NodeState("gs-c", vec3(50, 120, 0), vec3(0, 0, 0), False, t),
                ]
                stream.ingest_frame(
                    t,
                    nodes,
                    [
                        _bearing(nodes[0], "asset-a", target, 0.004),
                        _bearing(nodes[1], "asset-a", target, 0.004),
                        _bearing(nodes[2], "asset-a", target, 0.05),
                    ],
                    [TruthState("asset-a", target, vec3(4, 0, 0), t)],
                    target_metadata=[
                        TargetMetadata("asset-a", operator_label="Asset A", priority=3)
                    ],
                    safety_events=[
                        SafetyValidationResult(
                            f"safety-{step}",
                            "gs-c",
                            step % 2 == 0,
                            t,
                            violations=("min_agl",),
                            clamped=True,
                            reason="abort",
                        )
                    ],
                )
        finally:
            stream.close()
    finally:
        service.close()

    responses = _fuse_and_compare(tmp_path, recorded)
    last = responses[-1].frame
    assert last.tracks, "the two accepted nodes form a track"
    assert [event.event_id for event in last.safety_events] == ["safety-7"]
    assert last.target_metadata[0].operator_label == "Asset A"
    rejected = {(r.node_id, r.reason) for r in last.rejected_observations}
    assert ("gs-c", "bearing_noise_too_high") in rejected
