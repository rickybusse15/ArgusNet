from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from argusnet.v1 import world_model_pb2 as pb


def _require_benchmark_only(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--benchmark-only", default=False):
        pytest.skip("benchmark tests run only with --benchmark-only")


def _vec(x: float, y: float, z: float) -> pb.Vector3:
    return pb.Vector3(x_m=x, y_m=y, z_m=z)


def _request(observation_count: int) -> pb.IngestFrameRequest:
    request = pb.IngestFrameRequest(timestamp_s=1.0)
    for i in range(8):
        request.node_states.append(
            pb.NodeState(
                node_id=f"node-{i}",
                position=_vec(float(i), 0.0, 25.0),
                velocity=_vec(0.0, 0.0, 0.0),
                is_mobile=True,
                timestamp_s=1.0,
                health=1.0,
                sensor_type="optical",
                fov_half_angle_deg=180.0,
                max_range_m=2000.0,
            )
        )
    for i in range(observation_count):
        request.observations.append(
            pb.BearingObservation(
                node_id=f"node-{i % 8}",
                target_id=f"target-{i % 4}",
                origin=_vec(float(i), 0.0, 25.0),
                direction=_vec(1.0, 0.0, -0.1),
                bearing_std_rad=0.02,
                timestamp_s=1.0,
                confidence=0.95,
            )
        )
    return request


@pytest.mark.benchmark_fast
@pytest.mark.parametrize("observation_count", [16, 64, 256])
def test_ingest_frame_request_roundtrip(
    benchmark,
    request: pytest.FixtureRequest,
    observation_count: int,
) -> None:
    _require_benchmark_only(request)
    message = _request(observation_count)

    def run() -> pb.IngestFrameRequest:
        payload = message.SerializeToString()
        decoded = pb.IngestFrameRequest()
        decoded.ParseFromString(payload)
        return decoded

    benchmark(run)
