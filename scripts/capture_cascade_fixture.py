#!/usr/bin/env python3
"""Record build_observations' rejection cascade, decision by decision.

P4.5b ports the cascade. This drives the *reference* cascade over constructed
(node, target) pairs and records which gate rejected each one, so the Rust port
is compared against what Python does rather than against a reading of it.

Rather than reimplement the gate sequence here -- the mistake that would make
the fixture agree with the port and both disagree with the reference -- this
calls `build_observations` itself and reads the rejection reasons back out of
the ObservationBatch it returns.

Usage:
    python3 scripts/capture_cascade_fixture.py --out tests/fixtures/cascade_decisions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.core.types import TruthState  # noqa: E402
from argusnet.simulation.sim import (  # noqa: E402
    REJECT_BUILDING_OCCLUSION,
    REJECT_DROPOUT,
    REJECT_LOW_ELEVATION,
    REJECT_OBJECT_OCCLUSION,
    REJECT_OUT_OF_COVERAGE,
    REJECT_OUT_OF_RANGE,
    REJECT_OUTSIDE_FOV,
    REJECT_TERRAIN_OCCLUSION,
    REJECT_VEGETATION_OCCLUSION,
    REJECT_WALL_OCCLUSION,
    SimNode,
    build_observations,
)
from argusnet.world.terrain import TerrainModel  # noqa: E402


def exact(value: float) -> str:
    """Render a float as a string so it survives the trip into Rust.

    `serde_json`'s parser is off by one ULP on some literals, which at a gate
    boundary is a different decision. Established while porting the sensor
    error model; not repeating it here.
    """
    return repr(float(value))


def _gate_of(reason: str, detail: str) -> str:
    """Name the gate that produced a rejection, from the reason and detail.

    Reason strings are not unique to a gate. This maps each rejection back to
    the gate that made it, so gate ORDER is testable rather than only the
    final tally. Derived from the reference's own detail text; a detail that
    stops matching should fail loudly rather than silently mislabel.
    """
    if reason == "out_of_range":
        return "1_range"
    if reason == "outside_fov":
        return "2_fov"
    if reason == "dropout" and detail.startswith("Observation dropped with p="):
        return "3_dropout"
    if reason == "low_elevation":
        return "4_elevation"
    if detail.startswith("LOS blocked by"):
        return "5_line_of_sight"
    if detail.startswith("Probabilistic detection rejected"):
        return "6_detection"
    raise AssertionError(f"unrecognised rejection: reason={reason!r} detail={detail!r}")


def static_node(
    node_id: str,
    position,
    *,
    max_range_m: float,
    fov_half_angle_deg: float,
    dropout_probability: float,
    is_mobile: bool = False,
) -> SimNode:
    pos = np.array(position, dtype=float)

    def trajectory(_t, _p=pos):
        return _p.copy(), np.zeros(3)

    return SimNode(
        node_id=node_id,
        is_mobile=is_mobile,
        bearing_std_rad=0.01,
        dropout_probability=dropout_probability,
        max_range_m=max_range_m,
        trajectory=trajectory,
        fov_half_angle_deg=fov_half_angle_deg,
    )


def aimed_node(
    node_id: str,
    position,
    direction,
    *,
    max_range_m: float,
    fov_half_angle_deg: float,
) -> SimNode:
    """A node with an explicit look direction, so the FOV gate is reachable.

    A static node has no velocity, so the reference resolves no look direction
    and skips the FOV gate entirely. Supplying `sensor_direction_fn` is the only
    way to exercise gate 2 without giving the node a trajectory.
    """
    pos = np.array(position, dtype=float)
    unit = np.array(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)

    def trajectory(_t, _p=pos):
        return _p.copy(), np.zeros(3)

    return SimNode(
        node_id=node_id,
        is_mobile=False,
        bearing_std_rad=0.01,
        dropout_probability=0.0,
        max_range_m=max_range_m,
        trajectory=trajectory,
        fov_half_angle_deg=fov_half_angle_deg,
        sensor_direction_fn=lambda _t, _d=unit: _d.copy(),
    )


def truth_at(target_id: str, position) -> TruthState:
    return TruthState(
        target_id=target_id,
        position=np.array(position, dtype=float),
        velocity=np.zeros(3),
        timestamp_s=0.0,
    )


def capture() -> dict:
    cases: list[dict] = []

    # Flat ground at zero, so the line-of-sight gate turns only on the geometry
    # each case sets up rather than on an incidental ridge or basin.
    flat = TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=0.0,
        slope_x_m_per_m=0.0,
        slope_y_m_per_m=0.0,
        wave_amplitude_m=0.0,
        ridge_amplitude_m=0.0,
        basin_depth_m=0.0,
    )

    def record(
        name: str,
        note: str,
        nodes,
        truths,
        *,
        seed: int,
        min_elevation_deg: float = 0.5,
        terrain=None,
    ):
        batch = build_observations(
            rng=np.random.default_rng(0),
            nodes=nodes,
            truths=truths,
            timestamp_s=0.0,
            terrain=flat if terrain is None else terrain,
            min_elevation_deg=min_elevation_deg,
            seed=seed,
        )
        rejections = [
            {
                "node_id": r.node_id,
                "target_id": r.target_id,
                "reason": r.reason,
                # The reason alone is ambiguous: gate 3 (dropout) and gate 6
                # (probabilistic detection) both report "dropout", and gate 6
                # reports "vegetation_occlusion" when transmittance dominates,
                # which gate 5 also reports. Without the gate, a port that ran
                # the two in the wrong order would still match on reasons.
                "gate": _gate_of(r.reason, r.detail),
            }
            for r in batch.generation_rejections
        ]
        cases.append(
            {
                "name": name,
                "note": note,
                "seed": seed,
                "min_elevation_deg": exact(min_elevation_deg),
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "position": [exact(v) for v in n.trajectory(0.0)[0]],
                        "max_range_m": exact(n.max_range_m),
                        "fov_half_angle_deg": exact(n.fov_half_angle_deg),
                        "dropout_probability": exact(n.dropout_probability),
                        "is_mobile": bool(n.is_mobile),
                    }
                    for n in nodes
                ],
                "truths": [
                    {"target_id": t.target_id, "position": [exact(v) for v in t.position]}
                    for t in truths
                ],
                "expected": {
                    "attempted": int(batch.attempted_count),
                    "accepted": len(batch.observations),
                    "rejection_counts": dict(sorted(batch.rejection_counts.items())),
                    "rejected_by_target": dict(sorted(batch.rejected_by_target.items())),
                    "accepted_by_target": dict(sorted(batch.accepted_by_target.items())),
                    "rejections": rejections,
                },
            }
        )

    observer = (0.0, 0.0, 100.0)

    # A ridge centred between the gate-5 case's observer and target, tall
    # enough to break their line of sight.
    RIDGE = TerrainModel(
        ground_plane_m=0.0,
        base_elevation_m=0.0,
        slope_x_m_per_m=0.0,
        slope_y_m_per_m=0.0,
        wave_amplitude_m=0.0,
        ridge_amplitude_m=180.0,
        ridge_center_x_m=0.0,
        ridge_center_y_m=0.0,
        ridge_radius_m=200.0,
        basin_depth_m=0.0,
    )

    # Range gate, both bounds: a target beyond max_range and one coincident
    # with the node (range < 1e-6), which the same gate catches.
    record(
        "range_gate_both_bounds",
        "one target far beyond max range, one coincident with the node origin",
        [
            static_node(
                "n1", observer, max_range_m=500.0, fov_half_angle_deg=180.0, dropout_probability=0.0
            )
        ],
        [truth_at("far", (5000.0, 0.0, 100.0)), truth_at("coincident", observer)],
        seed=11,
    )

    # FOV gate. fov_half_angle_deg < 179 makes the sensor directional, but with
    # no velocity and no direction function the reference resolves no look
    # direction and skips the gate -- worth pinning, since it is a silent skip.
    record(
        "directional_sensor_without_look_direction",
        "fov_half_angle < 179 but a static node resolves no look direction, so the gate is skipped",
        [
            static_node(
                "n1", observer, max_range_m=5000.0, fov_half_angle_deg=30.0, dropout_probability=0.0
            )
        ],
        [truth_at("t1", (100.0, 0.0, 90.0)), truth_at("t2", (0.0, 400.0, 50.0))],
        seed=12,
    )

    # Dropout gate at p=1: every pair rejected, and rejected there rather than
    # at the elevation gate below it.
    record(
        "dropout_certain",
        "dropout_probability 1.0 rejects every pair at gate 3",
        [
            static_node(
                "n1",
                observer,
                max_range_m=5000.0,
                fov_half_angle_deg=180.0,
                dropout_probability=1.0,
            )
        ],
        [truth_at("t1", (300.0, 0.0, 90.0)), truth_at("t2", (0.0, 300.0, 95.0))],
        seed=13,
    )

    # The ordering case that matters: a target that would fail the elevation
    # gate, behind a certain dropout. Python reports dropout, not low_elevation.
    record(
        "dropout_precedes_elevation",
        "a target that would also fail elevation is reported as dropout, proving gate order",
        [
            static_node(
                "n1",
                observer,
                max_range_m=5000.0,
                fov_half_angle_deg=180.0,
                dropout_probability=1.0,
            )
        ],
        [truth_at("below", (3000.0, 0.0, 99.0))],
        seed=14,
    )

    # The same target with dropout off, to show the elevation gate does reject
    # it -- so the case above is genuinely about ordering, not about the target.
    record(
        "elevation_gate_alone",
        "the same geometry with dropout disabled, rejected at gate 4 instead",
        [
            static_node(
                "n1",
                observer,
                max_range_m=5000.0,
                fov_half_angle_deg=180.0,
                dropout_probability=0.0,
            )
        ],
        [truth_at("below", (3000.0, 0.0, 99.0))],
        seed=14,
    )

    # Mobile nodes take a different elevation floor from the constants.
    record(
        "mobile_elevation_floor",
        "a mobile node uses sensor_cfg.mobile_min_elevation_deg, not min_elevation_deg",
        [
            static_node(
                "drone",
                observer,
                max_range_m=5000.0,
                fov_half_angle_deg=180.0,
                dropout_probability=0.0,
                is_mobile=True,
            )
        ],
        [truth_at("below", (3000.0, 0.0, 99.0)), truth_at("above", (100.0, 0.0, 10.0))],
        seed=15,
    )

    # Several nodes and targets, so per-node and per-target tallies are exercised.
    record(
        "multi_node_multi_target",
        "three nodes and four targets, mixing accepted and rejected pairs",
        [
            static_node(
                "n1", observer, max_range_m=800.0, fov_half_angle_deg=180.0, dropout_probability=0.0
            ),
            static_node(
                "n2",
                (1000.0, 0.0, 120.0),
                max_range_m=400.0,
                fov_half_angle_deg=180.0,
                dropout_probability=0.0,
            ),
            static_node(
                "n3",
                (0.0, 1000.0, 80.0),
                max_range_m=5000.0,
                fov_half_angle_deg=180.0,
                dropout_probability=0.5,
            ),
        ],
        [
            truth_at("a", (200.0, 0.0, 10.0)),
            truth_at("b", (1100.0, 0.0, 20.0)),
            truth_at("c", (0.0, 900.0, 5.0)),
            truth_at("d", (4000.0, 4000.0, 0.0)),
        ],
        seed=16,
    )

    # Gate 2: an aimed sensor with one target inside the cone and one outside.
    record(
        "fov_gate_with_look_direction",
        "an aimed 30-degree sensor: one target on axis, one at 90 degrees off it",
        [
            aimed_node(
                "aimed", observer, (1.0, 0.0, 0.0), max_range_m=5000.0, fov_half_angle_deg=30.0
            )
        ],
        [truth_at("on_axis", (500.0, 0.0, 100.0)), truth_at("off_axis", (0.0, 500.0, 100.0))],
        seed=17,
    )

    # Gate 5: a ridge between a low observer and a low target blocks the line of
    # sight, which no other case reaches.
    record(
        "terrain_occlusion",
        "a ridge blocks the line of sight; the target sits high enough to clear "
        "the elevation gate first, so this reaches gate 5 rather than gate 4",
        [
            static_node(
                "ground",
                (-400.0, 0.0, 5.0),
                max_range_m=5000.0,
                fov_half_angle_deg=180.0,
                dropout_probability=0.0,
            )
        ],
        [truth_at("behind_ridge", (400.0, 0.0, 60.0))],
        seed=18,
        terrain=RIDGE,
    )

    return {
        "source": "src/argusnet/simulation/sim.py::build_observations",
        "note": (
            "Rejection-cascade decisions read back from the ObservationBatch the "
            "reference itself produced, so the fixture cannot agree with the port "
            "by sharing its author's misreading. Reason strings are the wire "
            "strings; note that object and building occlusion are the same string."
        ),
        "reason_aliasing": {
            "object_occlusion_is_building_occlusion": REJECT_OBJECT_OCCLUSION
            == REJECT_BUILDING_OCCLUSION,
            "distinct_reason_strings": sorted(
                {
                    REJECT_OUT_OF_RANGE,
                    REJECT_OUTSIDE_FOV,
                    REJECT_DROPOUT,
                    REJECT_LOW_ELEVATION,
                    REJECT_TERRAIN_OCCLUSION,
                    REJECT_BUILDING_OCCLUSION,
                    REJECT_WALL_OCCLUSION,
                    REJECT_VEGETATION_OCCLUSION,
                    REJECT_OUT_OF_COVERAGE,
                    REJECT_OBJECT_OCCLUSION,
                }
            ),
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    total = sum(c["expected"]["attempted"] for c in document["cases"])
    print(f"wrote {args.out}: {len(document['cases'])} cases, {total} pairs evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
