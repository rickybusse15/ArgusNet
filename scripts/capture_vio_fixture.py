#!/usr/bin/env python3
"""Record EKFVIO's prediction step, state and covariance, step by step.

P4.4's prediction step is deterministic given buffered IMU, so it can be
compared value-for-value -- the approach that worked for P4.2a and P4.5a.

The UPDATE step is deliberately not recorded as a parity target. Two of its
behaviours are things P4.4 requires changing (see
docs/migration/VIO_CONTRACT.md): a scalar fallback that manufactures a
measurement from the feature count, and an out-of-order frame that leaves the
feature store stale. A fixture recording those as "expected" would bless them.
So every case here drives the PREDICTION path only -- images carrying no
matched features, so the update is a no-op -- and the divergent behaviours get
contract tests on the Rust side instead.

Usage:
    python3 scripts/capture_vio_fixture.py --out tests/fixtures/vio_prediction.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.localization.vio import EKFVIO, VisualFeature  # noqa: E402
from argusnet.sensing.imu import IMUMeasurement  # noqa: E402


def exact(value: float) -> str:
    """Render a float as a string so it survives the trip into Rust.

    `serde_json`'s parser is off by one ULP on some literals. In a filter that
    integrates over many steps, an ULP at the input is not an ULP at the
    output. Established while porting the sensor error model.
    """
    return repr(float(value))


def imu(t: float, gyro, accel) -> IMUMeasurement:
    return IMUMeasurement(
        angular_velocity=np.array(gyro, dtype=float),
        linear_acceleration=np.array(accel, dtype=float),
        timestamp_s=t,
    )


def capture() -> dict:
    cases: list[dict] = []

    def record(name: str, note: str, initial_position, measurements, frame_times):
        """Drive the reference and record its state after each image frame.

        Features are unique per frame, so no feature ever matches across
        frames and the update step cannot fire. What is recorded is the
        prediction path alone.
        """
        vio = EKFVIO()
        vio.initialize(np.array(initial_position, dtype=float), 0.0)
        steps = []
        pending = list(measurements)
        for counter, frame_t in enumerate(frame_times, start=1):
            # Every measurement still pending is handed over, including any
            # beyond this frame. The reference's own guards decide what to
            # consume -- a fixture that pre-filtered them would make those
            # guards untestable, which is exactly what an earlier version did.
            fed = list(pending)
            pending = []
            for m in fed:
                vio.process_imu(m)
            # Feature ids unique to this frame: nothing matches, update is a no-op.
            features = [
                VisualFeature(feature_id=counter * 1000 + k, u=1.0, v=2.0) for k in range(3)
            ]
            state = vio.process_image(features, timestamp_s=frame_t)
            assert state is not None, "initialised estimator must return a state"
            assert state.tracked_feature_count == 0, (
                "a matched feature would mean the update fired; "
                "this fixture must record prediction only"
            )
            steps.append(
                {
                    "timestamp_s": exact(frame_t),
                    "imu_fed": [
                        {
                            "timestamp_s": exact(m.timestamp_s),
                            "angular_velocity": [exact(v) for v in m.angular_velocity],
                            "linear_acceleration": [exact(v) for v in m.linear_acceleration],
                        }
                        for m in fed
                    ],
                    "expected": {
                        "state": [exact(v) for v in vio._x],  # noqa: SLF001
                        # Full matrix, row-major. The diagonal alone is blind
                        # to the Jacobian's off-diagonal terms -- flipping the
                        # sign of d(rot)/d(gyro_bias) left every diagonal entry
                        # unchanged and went undetected.
                        "covariance": [exact(v) for v in vio._P.ravel()],  # noqa: SLF001
                        "position": [exact(v) for v in state.position],
                        "velocity": [exact(v) for v in state.velocity],
                        # What sim.py actually consumes as its confidence signal.
                        "covariance_pos_trace": exact(np.trace(state.covariance[6:9, 6:9])),
                    },
                }
            )
        return steps

    def add(name, note, initial_position, measurements, frame_times):
        cases.append(
            {
                "name": name,
                "note": note,
                "initial_position": [exact(v) for v in initial_position],
                "steps": record(name, note, initial_position, measurements, frame_times),
            }
        )

    # Free fall in the estimator's terms: an IMU reading zero specific force
    # means gravity is the only world acceleration.
    add(
        "no_imu_constant_velocity",
        "no IMU at all, so each frame takes the single constant-velocity step",
        (0.0, 0.0, 100.0),
        [],
        [0.1, 0.2, 0.35, 0.5],
    )

    # A stationary platform: specific force exactly cancels gravity.
    add(
        "stationary_gravity_cancelled",
        "accelerometer reads +g, so world acceleration is zero and nothing moves",
        (10.0, -5.0, 50.0),
        [imu(0.02 * k, (0.0, 0.0, 0.0), (0.0, 0.0, 9.80665)) for k in range(1, 26)],
        [0.1, 0.2, 0.3, 0.5],
    )

    # Constant turn: gyro drives the rotation state, which rotates the
    # accelerometer into the world frame differently each step.
    add(
        "constant_turn",
        "steady yaw rate with a forward specific force; exercises the rotation propagation",
        (0.0, 0.0, 80.0),
        [imu(0.02 * k, (0.0, 0.0, 0.35), (1.2, 0.0, 9.80665)) for k in range(1, 26)],
        [0.1, 0.25, 0.4, 0.5],
    )

    # Irregular IMU arrival, including a measurement exactly on a frame
    # boundary and one beyond the frame (which must be skipped, not consumed).
    add(
        "irregular_and_boundary_imu",
        "uneven intervals, a sample exactly on a frame timestamp, and one past it",
        (1.0, 2.0, 30.0),
        [
            imu(0.013, (0.01, -0.02, 0.03), (0.5, -0.3, 9.9)),
            imu(0.1, (0.0, 0.0, 0.1), (0.2, 0.2, 9.7)),  # exactly on frame 1
            imu(0.157, (0.05, 0.0, 0.0), (-0.4, 0.1, 9.85)),
            imu(0.29, (0.0, 0.03, -0.02), (0.0, 0.6, 9.75)),
            imu(0.95, (0.0, 0.0, 0.0), (0.0, 0.0, 9.80665)),  # beyond every frame
        ],
        [0.1, 0.3, 0.45],
    )

    # A gap with no IMU: the constant-velocity fallback must cover the whole
    # interval, not part of it.
    add(
        "imu_gap",
        "IMU covers the first frame only; later frames fall back to constant velocity",
        (0.0, 0.0, 60.0),
        [imu(0.02 * k, (0.0, 0.0, 0.0), (2.0, 0.0, 9.80665)) for k in range(1, 6)],
        [0.1, 0.4, 0.9],
    )

    # dt_total == 0 is NOT caught by the `< 0` guard, so a repeated frame
    # timestamp proceeds with a zero-length interval. Pinned so a port that
    # rejects it as invalid fails here.
    add(
        "repeated_frame_timestamp",
        "the same frame time twice; dt_total is zero, which the < 0 guard does not reject",
        (5.0, 5.0, 40.0),
        [imu(0.02 * k, (0.0, 0.0, 0.1), (1.0, 0.0, 9.80665)) for k in range(1, 11)],
        [0.1, 0.2, 0.2, 0.3],
    )

    # An IMU sample landing exactly on a previous frame's timestamp must be
    # skipped by `meas.timestamp_s <= t_prev`, not consumed with dt = 0.
    add(
        "imu_exactly_on_previous_frame_time",
        "a sample at exactly the prior frame time, which the <= guard must skip",
        (0.0, 0.0, 20.0),
        [
            imu(0.05, (0.0, 0.0, 0.0), (1.0, 0.0, 9.80665)),
            imu(0.10, (0.0, 0.0, 0.0), (1.0, 0.0, 9.80665)),
            imu(0.10, (0.0, 0.0, 0.0), (5.0, 5.0, 9.80665)),  # duplicate of the frame time
            imu(0.15, (0.0, 0.0, 0.0), (1.0, 0.0, 9.80665)),
        ],
        [0.1, 0.2],
    )

    return {
        "source": "src/argusnet/localization/vio.py::EKFVIO (prediction path)",
        "note": (
            "State and covariance after each image frame, driven so no feature "
            "ever matches and the update step cannot fire. The update's scalar "
            "fallback and out-of-order handling are NOT recorded here: P4.4 "
            "requires changing both, so a parity fixture would bless them. See "
            "docs/migration/VIO_CONTRACT.md."
        ),
        "state_layout": (
            "rotation(0:3) velocity(3:6) position(6:9) accel_bias(9:12) gyro_bias(12:15)"
        ),
        "gravity_world": [exact(0.0), exact(0.0), exact(9.80665)],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    total = sum(len(c["steps"]) for c in document["cases"])
    print(f"wrote {args.out}: {len(document['cases'])} cases, {total} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
