#!/usr/bin/env python3
"""Record EKFVIO's optical-flow update -- both where the port agrees and where
it deliberately does not.

This fixture has an unusual shape, and the shape is the point. Every other
parity fixture in this migration records the reference as the TARGET. Two of
this estimator's update behaviours are ones P4.4 requires changing (see
docs/migration/VIO_CONTRACT.md), so recording them as targets would bless
them.

Cases are therefore tagged:

  agreement  -- the port must reproduce these values.
  divergence -- the port must NOT reproduce these; the recording exists so the
                change is provable rather than asserted, and so a regression
                back to the old behaviour is detectable.

Usage:
    python3 scripts/capture_vio_update_fixture.py --out tests/fixtures/vio_update.json
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
    return repr(float(value))


def build(initial_position, initial_velocity_imu=None):
    """A filter seeded at a position, optionally given velocity via one IMU step."""
    vio = EKFVIO()
    vio.initialize(np.array(initial_position, dtype=float), 0.0)
    if initial_velocity_imu is not None:
        accel, until = initial_velocity_imu
        vio.process_imu(
            IMUMeasurement(
                angular_velocity=np.zeros(3),
                linear_acceleration=np.array(accel, dtype=float),
                timestamp_s=until,
            )
        )
    return vio


def snapshot(vio) -> dict:
    return {
        "state": [exact(v) for v in vio._x],  # noqa: SLF001
        "covariance": [exact(v) for v in vio._P.ravel()],  # noqa: SLF001
        "position_covariance_trace": exact(np.trace(vio._P[6:9, 6:9])),  # noqa: SLF001
        "feature_store_ids": sorted(int(i) for i in vio._prev_feature_ids),  # noqa: SLF001
    }


def capture() -> dict:
    cases: list[dict] = []

    def record(name, kind, note, first_frame, second_frame, *, accel=None):
        """Run two frames and record the state before and after the second."""
        vio = build((0.0, 0.0, 100.0), initial_velocity_imu=(accel, 0.5) if accel else None)
        t1, feats1 = first_frame
        vio.process_image(feats1, timestamp_s=t1)
        before = snapshot(vio)
        t2, feats2 = second_frame
        state = vio.process_image(feats2, timestamp_s=t2)
        after = snapshot(vio)
        cases.append(
            {
                "name": name,
                "kind": kind,
                "note": note,
                "first_frame": {
                    "timestamp_s": exact(t1),
                    "features": [
                        {"feature_id": f.feature_id, "u": exact(f.u), "v": exact(f.v)}
                        for f in feats1
                    ],
                },
                "second_frame": {
                    "timestamp_s": exact(t2),
                    "features": [
                        {"feature_id": f.feature_id, "u": exact(f.u), "v": exact(f.v)}
                        for f in feats2
                    ],
                },
                "accel_for_velocity": None if accel is None else [exact(v) for v in accel],
                "before_second_frame": before,
                "reference_after": after,
                "reference_tracked_feature_count": (
                    None if state is None else int(state.tracked_feature_count)
                ),
            }
        )

    def feats(spec):
        return [VisualFeature(feature_id=i, u=float(u), v=float(v)) for i, u, v in spec]

    # Eight features, not four. The reference's scalar fallback requires
    # matched >= min_features, and min_features defaults to 6 -- with four
    # features the fallback never fires and the divergence case silently
    # recorded a no-op. Eight reproduces the defect at the DEFAULT config,
    # which is the one sim.py uses.
    base = [
        (1, 100.0, 200.0),
        (2, 150.0, 220.0),
        (3, 200.0, 180.0),
        (4, 250.0, 240.0),
        (5, 300.0, 160.0),
        (6, 340.0, 260.0),
        (7, 380.0, 140.0),
        (8, 420.0, 280.0),
    ]

    # AGREEMENT: features move a little, all pass the gate, batch update applies.
    record(
        "batch_update_all_features_pass",
        "agreement",
        "small consistent flow; every feature passes the 3-sigma gate and the "
        "batch update applies. The port must reproduce this exactly.",
        (0.5, feats(base)),
        (0.6, feats([(i, u + 0.4, v + 0.25) for i, u, v in base])),
        accel=(2.0, 1.0, 9.80665),
    )

    # AGREEMENT: no feature carries over, so nothing is applied either way.
    record(
        "no_matched_features",
        "agreement",
        "disjoint feature ids, so nothing matches and no update occurs in either implementation.",
        (0.5, feats(base)),
        (0.6, feats([(i + 900, u, v) for i, u, v in base])),
        accel=(2.0, 1.0, 9.80665),
    )

    # DIVERGENCE: every feature is a wild outlier. The reference falls back to
    # a measurement synthesised from the feature count.
    record(
        "all_features_gated_out",
        "divergence",
        "every feature jumps 9e4 px and fails the gate. The reference "
        "fabricates y = (matched/(min_features*2) - 0.5) * 0.1, moves the "
        "position and SHRINKS the covariance. The port must apply nothing and "
        "leave prediction's covariance standing.",
        (0.5, feats(base)),
        (0.6, feats([(i, u + 9e4, v + 9e4) for i, u, v in base])),
        accel=(2.0, 1.0, 9.80665),
    )

    # DIVERGENCE: an out-of-order frame. The reference returns before
    # refreshing the feature store, so the store keeps the OLD ids.
    record(
        "out_of_order_frame",
        "divergence",
        "the second frame is older than the first. The reference drops it and "
        "leaves the feature store holding the first frame's ids, so the next "
        "frame matches against a two-frame-old store. The port refreshes the "
        "store on every frame it sees.",
        (1.0, feats(base)),
        (0.5, feats([(i + 500, u, v) for i, u, v in base])),
        accel=(2.0, 1.0, 9.80665),
    )

    return {
        "source": "src/argusnet/localization/vio.py::EKFVIO.process_image (update path)",
        "note": (
            "Cases tagged 'agreement' are parity targets. Cases tagged "
            "'divergence' record behaviour P4.4 requires CHANGING -- they are "
            "evidence that the change is real, not targets. See "
            "docs/migration/VIO_CONTRACT.md."
        ),
        "mahalanobis_gate": exact(11.8),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    kinds = {}
    for case in document["cases"]:
        kinds[case["kind"]] = kinds.get(case["kind"], 0) + 1
    print(f"wrote {args.out}: {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
