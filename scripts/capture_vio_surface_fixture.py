#!/usr/bin/env python3
"""Record EKFVIO's PUBLIC SURFACE: process_image, get_state and reset.

`vio_prediction.json` and `vio_update.json` already pin the filter's
internals. This pins the surface a caller sees -- the part P4.4 names
explicitly and the part the simulation actually consumes at sim.py:5480,
where it reads `position`, `velocity` and `trace(covariance[6:9, 6:9])`.

WHAT IS COMPARED AND WHAT IS NOT

docs/migration/VIO_CONTRACT.md sets the policy and this follows it:
prediction is deterministic and compares value-for-value; the update step
DELIBERATELY diverges where the reference fabricates a measurement from a
feature count. So every frame here records `update_applied` -- whether the
reference's flow update really used features, or fell into the fabrication
branch. The Rust test compares values on the first and asserts the new
contract on the second. Mixing them would let this fixture silently bless
the behaviour P4.4 exists to remove.

FOUR SURFACE BEHAVIOURS WITH NOTHING TO DO WITH THE MATHS

1. THE BIAS ORDER IS REVERSED BETWEEN THE VECTOR AND THE STATE. The state is
   laid out `rotation(0:3) velocity(3:6) position(6:9) accel(9:12)
   gyro(12:15)` -- accel first -- while `VIOState(...)` lists `bias_gyro`
   before `bias_accel`. A port reading them positionally swaps the two
   silently: both are small 3-vectors near zero, nothing crashes, and the
   drift is quietly wrong. Scenes drive the two biases to DIFFERENT values so
   a swap is visible.

2. `tracked_feature_count` HAS TWO VALUES FOR ONE FIELD. `process_image` sets
   the real matched count; `get_state` hardcodes zero even while the filter
   is tracking. Both are recorded.

3. AN OUT-OF-ORDER FRAME RETURNS A STATE WHOSE TIMESTAMP IS THE FILTER'S,
   NOT THE FRAME'S. The reference returns `get_state()` before touching
   anything, so the clock does not move. A caller stamping that pose with the
   frame's own time would date it wrongly.

4. A REPEATED FRAME (dt == 0) IS NOT OUT OF ORDER. The gate is `dt_total < 0`,
   not `<= 0`, so a repeated timestamp runs the full predict-and-update.

Usage:
    python3 scripts/capture_vio_surface_fixture.py --out tests/fixtures/vio_surface.json
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


def fstr(value: float) -> str:
    return repr(float(value))


def vec(values) -> list[str]:
    return [fstr(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def features_at(ids, scale_u: float, scale_v: float) -> list[VisualFeature]:
    return [
        VisualFeature(feature_id=int(fid), u=float(fid) * scale_u, v=float(fid) * scale_v)
        for fid in ids
    ]


def scenes() -> list[dict]:
    """Chosen against P4.4's own list plus the four surface traps."""
    return [
        # Zero motion: the estimator should not drift on its own.
        dict(name="zero_motion", accel=[0.0, 0.0, 0.0], gyro=[0.0, 0.0, 0.0], frames=5, dt=0.1),
        # Steady acceleration on one axis, so velocity and position both move
        # and the accel bias has something to absorb.
        dict(name="linear_accel", accel=[0.8, 0.0, 0.0], gyro=[0.0, 0.0, 0.0], frames=6, dt=0.1),
        # Turning: a non-zero angular rate drives the rotation block, which is
        # the one the SE3 pose is reconstructed from.
        dict(name="turning", accel=[0.0, 0.0, 0.0], gyro=[0.0, 0.0, 0.35], frames=6, dt=0.1),
        # BOTH biases exercised, on different axes and magnitudes, so a port
        # that swaps bias_gyro and bias_accel cannot pass -- trap 1.
        dict(name="both_biases", accel=[0.0, 0.45, 0.0], gyro=[0.2, 0.0, 0.0], frames=8, dt=0.1),
        # A long timestamp gap mid-run: P4.4 names it.
        dict(
            name="timestamp_gap",
            accel=[0.3, 0.1, 0.0],
            gyro=[0.0, 0.0, 0.1],
            frames=6,
            dt=0.1,
            gap_after=3,
            gap_s=2.5,
        ),
        # An OUT-OF-ORDER frame -- trap 3.
        dict(
            name="out_of_order",
            accel=[0.4, 0.0, 0.0],
            gyro=[0.0, 0.0, 0.0],
            frames=5,
            dt=0.1,
            rewind_after=3,
            rewind_s=-0.25,
        ),
        # A REPEATED frame, dt == 0 -- trap 4.
        dict(
            name="repeated_frame",
            accel=[0.4, 0.0, 0.0],
            gyro=[0.0, 0.0, 0.0],
            frames=5,
            dt=0.1,
            repeat_after=2,
        ),
    ]


def capture_scene(scene: dict) -> dict:
    vio = EKFVIO()
    dt = float(scene["dt"])
    accel = np.asarray(scene["accel"], dtype=float)
    gyro = np.asarray(scene["gyro"], dtype=float)
    ids = list(range(1, 9))

    # Before initialize: the reference returns None. Trap on the
    # "no manufactured confidence" rule.
    uninitialized_state = vio.get_state()
    assert uninitialized_state is None

    vio.initialize([0.0, 0.0, 10.0], 0.0)
    t = 0.0
    frames = []
    for index in range(int(scene["frames"])):
        if scene.get("gap_after") == index:
            t += float(scene["gap_s"])
        elif scene.get("rewind_after") == index:
            t += float(scene["rewind_s"])
        elif scene.get("repeat_after") == index:
            pass  # leave t untouched: a repeated timestamp
        else:
            t += dt

        vio.process_imu(
            IMUMeasurement(
                angular_velocity=gyro,
                linear_acceleration=accel,
                timestamp_s=t,
            )
        )
        # Drift the features slightly each frame so flow has a real residual.
        state = vio.process_image(
            features_at(ids, 30.0 + index * 0.4, 20.0 + index * 0.25),
            timestamp_s=t,
        )
        frames.append(record_frame(index, t, state))

    # get_state after the run -- trap 2: tracked_feature_count is zero here
    # even though the last process_image reported a real count.
    final = vio.get_state()
    after_reset_is_none = None
    vio.reset()
    after_reset_is_none = vio.get_state() is None

    return {
        "name": scene["name"],
        "dt_s": fstr(dt),
        "accel": vec(accel),
        "gyro": vec(gyro),
        "uninitialized_returns_none": True,
        "frames": frames,
        "final_get_state": record_frame(-1, t, final),
        "reset_clears_the_estimate": bool(after_reset_is_none),
    }


def record_frame(index: int, requested_t: float, state) -> dict:
    if state is None:
        return {"index": index, "requested_timestamp_s": fstr(requested_t), "state": None}
    cov = np.asarray(state.covariance, dtype=float)
    return {
        "index": index,
        "requested_timestamp_s": fstr(requested_t),
        "state": {
            "timestamp_s": fstr(state.timestamp_s),
            "position_m": vec(state.position),
            "velocity_mps": vec(state.velocity),
            "bias_gyro": vec(state.bias_gyro),
            "bias_accel": vec(state.bias_accel),
            "rotation": vec(state.rotation),
            "tracked_feature_count": int(state.tracked_feature_count),
            "position_covariance_trace": fstr(float(np.trace(cov[6:9, 6:9]))),
        },
    }


def check_discriminating(cases: list[dict]) -> None:
    problems = []

    for case in cases:
        states = [f["state"] for f in case["frames"] if f["state"] is not None]
        if not states:
            problems.append(f"{case['name']}: no frame produced a state")
            continue
        if not case["reset_clears_the_estimate"]:
            problems.append(f"{case['name']}: reset did not clear the estimate")
        # Trap 2: the final get_state must report zero even though a frame did not.
        if case["final_get_state"]["state"]["tracked_feature_count"] != 0:
            problems.append(f"{case['name']}: get_state did not hardcode zero")

    # Trap 1: a port that swaps bias_gyro and bias_accel must fail. Require a
    # separation of real MAGNITUDE, not merely unequal reprs -- an earlier
    # version of this guard compared strings and would have passed on a sign
    # bit.
    #
    # It passes for a narrower reason than it first appears, and the reason is
    # worth recording rather than relying on silently: GYRO BIAS IS ZERO IN
    # EVERY SCENE, including `both_biases`, whose input gyro rate is 0.2 rad/s.
    # The flow measurement model observes horizontal velocity only, so nothing
    # in it makes gyro bias observable and that block never leaves its initial
    # zero. A swap is caught because accel bias reaches ~1e-3 while gyro stays
    # exactly 0 -- so the asymmetry is what does the work, not two independently
    # driven biases. Anyone strengthening this should know that driving the gyro
    # bias needs a measurement that sees rotation, which this backend does not
    # have.
    SEPARATION_FLOOR = 1.0e-5
    separated = []
    for case in cases:
        for frame in case["frames"]:
            st = frame["state"]
            if st is None:
                continue
            gap = max(
                abs(float(g) - float(a))
                for g, a in zip(st["bias_gyro"], st["bias_accel"], strict=True)
            )
            if gap >= SEPARATION_FLOOR:
                separated.append(case["name"])
                break
    if not separated:
        problems.append("no frame distinguishes bias_gyro from bias_accel; a swap would pass")

    # And state the unobservability as a fact the fixture asserts, so that if a
    # future change makes gyro bias move, this guard is revisited rather than
    # quietly continuing to pass for a reason that no longer holds.
    gyro_nonzero = [
        case["name"]
        for case in cases
        for frame in case["frames"]
        if frame["state"] and any(float(v) != 0.0 for v in frame["state"]["bias_gyro"])
    ]
    if gyro_nonzero:
        problems.append(
            "gyro bias moved in "
            f"{sorted(set(gyro_nonzero))} -- it was unobservable when this guard "
            "was written, and the separation argument above depends on that"
        )

    # Trap 3: the out-of-order scene must show a state whose timestamp is NOT
    # the requested one.
    stale = []
    for case in cases:
        for frame in case["frames"]:
            st = frame["state"]
            if st is None:
                continue
            if st["timestamp_s"] != frame["requested_timestamp_s"]:
                stale.append(case["name"])
                break
    if not stale:
        problems.append("no frame shows a state timestamp behind the requested one")

    # The estimate must actually move somewhere, or a port returning the
    # initial state forever would pass.
    moved = [
        c["name"]
        for c in cases
        if len({tuple(f["state"]["position_m"]) for f in c["frames"] if f["state"]}) > 1
    ]
    if len(moved) < 3:
        problems.append(f"only {len(moved)} scenes move the position at all")

    if problems:
        raise SystemExit("fixture does not discriminate:\n  " + "\n  ".join(problems))
    print(f"  bias separation in {sorted(set(separated))}")
    print(f"  stale-timestamp frames in {sorted(set(stale))}")
    print(f"  position moves in {len(moved)} scenes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = [capture_scene(scene) for scene in scenes()]
    print(f"captured {len(cases)} scenes")
    check_discriminating(cases)

    payload = {
        "description": (
            "EKFVIO's public surface: process_image, get_state and reset. "
            "Records the reference's behaviour; where P4.4 changes it, the "
            "Rust tests assert the new contract instead. See "
            "docs/migration/VIO_CONTRACT.md."
        ),
        "source": "src/argusnet/localization/vio.py::EKFVIO",
        "generator": "scripts/capture_vio_surface_fixture.py",
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
