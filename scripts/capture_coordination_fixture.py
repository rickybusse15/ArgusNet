#!/usr/bin/env python3
"""Record CoordinationManager: election, latency-queued claims, formations.

Three groups of cases, one per concern.

`elect_coordinator` is pure, so its cases are inputs and an elected id. Two of
them pin behaviour that reads like a bug and is reproduced deliberately: a
drone missing from `battery_states` scores a FULL fraction rather than being
skipped, and the running best starts at -1.0 while the running id starts at
`drone_ids[0]`, so when every fraction is below -1.0 the first id wins no
matter who actually holds the most energy.

`update_claimed`/`flush_messages` are stateful, so those cases are SEQUENCES of
calls with the whole shared state recorded after each one -- claimed cells and
every pending queue, queues in key order. A port that applied a drone's due
messages in the wrong order, or that dropped a queue once it drained, passes
every single-call check and still diverges here.

`formation_offsets` is pure but order-sensitive in its output, so its cases
record the offsets as an ORDERED list of (drone_id, [x, y]) pairs. The set has
to reach RANK 3 -- nine drones -- to pin the arithmetic at all: ranks 1 and 2
scale a float exactly, so a five-drone formation gives the same answer however
the factors are grouped, however odd the spacing. `check_discriminating`
recomputes every offset the wrong way and refuses to write a fixture whose
cases cannot tell the two groupings apart.

Usage:
    python3 scripts/capture_coordination_fixture.py --out tests/fixtures/coordination.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.planning.battery import BatteryState  # noqa: E402
from argusnet.planning.coordination import (  # noqa: E402
    CoordinationManager,
    CoordinationPolicy,
    SharedMissionState,
)


def exact(value: float) -> str:
    return repr(float(value))


def battery(remaining_wh: float) -> BatteryState:
    return BatteryState(remaining_wh=remaining_wh, timestamp_s=0.0)


# ----------------------------------------------------------------------
# Election
# ----------------------------------------------------------------------


def election_cases() -> list[dict]:
    cases: list[tuple[str, list[str], dict[str, float], float]] = [
        ("empty_list", [], {}, 500.0),
        ("single_drone", ["a"], {"a": 120.0}, 500.0),
        (
            "highest_fraction_wins",
            ["a", "b", "c"],
            {"a": 100.0, "b": 400.0, "c": 250.0},
            500.0,
        ),
        # Strict `>` keeps the first, so the tie goes to "b" purely because it
        # is listed first -- not because of anything about "b".
        ("tie_keeps_the_earlier_id", ["b", "a"], {"a": 300.0, "b": 300.0}, 500.0),
        ("tie_reversed_order", ["a", "b"], {"a": 300.0, "b": 300.0}, 500.0),
        # A drone with no battery state scores 1.0 and outranks a nearly-full
        # known drone.
        ("missing_state_scores_full", ["known", "unknown"], {"known": 499.0}, 500.0),
        ("missing_state_listed_first", ["unknown", "known"], {"known": 499.0}, 500.0),
        # Under a true fraction both are 1.0 and the first wins; the
        # `max(capacity, 1.0)` divisor makes them 0.25 and 0.5 instead, so
        # "large" wins. Dividing by the capacity would flip this case.
        ("capacity_below_one_wh", ["small", "large"], {"small": 0.25, "large": 0.5}, 0.5),
        ("zero_capacity", ["small", "large"], {"small": 0.25, "large": 0.5}, 0.0),
        # Every fraction below the -1.0 starting bar: "b" holds more energy and
        # still loses to the initial id.
        ("all_below_the_bar", ["a", "b"], {"a": -5000.0, "b": -1000.0}, 500.0),
        ("all_below_the_bar_reversed", ["b", "a"], {"a": -5000.0, "b": -1000.0}, 500.0),
        # Exactly at the bar: -500/500 == -1.0 is not > -1.0, so "x" is kept by
        # the initial assignment rather than by the comparison.
        ("exactly_at_the_bar", ["x", "y"], {"x": -500.0, "y": -500.0}, 500.0),
        # A fraction just above the bar does clear it.
        ("just_above_the_bar", ["x", "y"], {"x": -600.0, "y": -499.0}, 500.0),
        (
            "empty_battery_map",
            ["a", "b"],
            {},
            500.0,
        ),
    ]

    manager = CoordinationManager()
    out = []
    for name, drone_ids, wh, capacity in cases:
        states = {did: battery(v) for did, v in wh.items()}
        elected = manager.elect_coordinator(list(drone_ids), states, capacity)
        out.append(
            {
                "name": name,
                "drone_ids": list(drone_ids),
                "battery_states": [
                    {"drone_id": did, "remaining_wh": exact(v)} for did, v in wh.items()
                ],
                "battery_capacity_wh": exact(capacity),
                "elected": elected,
            }
        )
    return out


# ----------------------------------------------------------------------
# Claimed-cell updates under latency
# ----------------------------------------------------------------------


def snapshot(shared: SharedMissionState) -> dict:
    return {
        "claimed_cells": [
            {"drone_id": did, "cell": list(cell)}
            for did, cell in sorted(shared.claimed_cells.items())
        ],
        # Queue order is load-bearing: it is the order the drones first queued
        # anything, and a drained queue stays behind as an empty one.
        "pending": [
            {
                "drone_id": did,
                "messages": [
                    {
                        "cell": list(msg["cell"]),
                        "deliver_at_step": msg["deliver_at_step"],
                    }
                    for msg in messages
                ],
            }
            for did, messages in shared.pending_messages.items()
        ],
    }


MESSAGE_SEQUENCES: list[tuple[str, int, list[tuple]]] = [
    (
        "zero_latency_applies_immediately",
        0,
        [
            ("update", "a", (3, 4), 10),
            ("update", "b", (5, 6), 10),
            ("flush", 10),
            ("update", "a", (7, 8), 11),
        ],
    ),
    (
        "none_cell_is_a_no_op",
        0,
        [
            ("update", "a", None, 0),
            ("flush", 0),
        ],
    ),
    (
        "none_cell_is_a_no_op_under_latency",
        3,
        [
            ("update", "a", None, 0),
            ("flush", 9),
        ],
    ),
    (
        "queued_message_waits_for_its_step",
        2,
        [
            ("update", "a", (1, 1), 5),
            ("flush", 5),
            ("flush", 6),
            ("flush", 7),
        ],
    ),
    (
        "last_message_due_on_a_step_wins",
        2,
        [
            ("update", "a", (1, 1), 5),
            ("update", "a", (2, 2), 5),
            ("update", "a", (3, 3), 5),
            ("flush", 7),
        ],
    ),
    (
        "a_later_message_is_held_back",
        2,
        [
            ("update", "a", (1, 1), 5),
            ("update", "a", (2, 2), 6),
            ("flush", 7),
            ("flush", 8),
        ],
    ),
    (
        "queue_order_is_first_queued_first",
        4,
        [
            ("update", "c", (0, 0), 0),
            ("update", "a", (0, 1), 0),
            ("update", "c", (1, 0), 1),
            ("update", "b", (0, 2), 2),
            ("flush", 4),
            ("flush", 5),
            ("flush", 6),
        ],
    ),
    (
        "a_drained_queue_survives_as_an_empty_queue",
        1,
        [
            ("update", "a", (1, 1), 0),
            ("flush", 5),
            ("flush", 6),
        ],
    ),
    (
        "negative_latency_delivers_at_the_next_flush",
        -3,
        [
            ("update", "a", (7, 7), 10),
            ("flush", 10),
        ],
    ),
    (
        "latency_one_delivers_on_the_following_step",
        1,
        [
            ("update", "a", (1, 2), 0),
            ("flush", 0),
            ("flush", 1),
        ],
    ),
    (
        "interleaved_drones_under_latency",
        2,
        [
            ("update", "a", (1, 1), 0),
            ("update", "b", (2, 2), 1),
            ("update", "a", (3, 3), 2),
            ("flush", 2),
            ("flush", 3),
            ("flush", 4),
        ],
    ),
    (
        "negative_cells_round_trip",
        0,
        [
            ("update", "a", (-4, -9), 0),
        ],
    ),
]


def message_cases() -> list[dict]:
    manager = CoordinationManager()
    out = []
    for name, latency, steps in MESSAGE_SEQUENCES:
        shared = SharedMissionState(message_latency_steps=latency)
        recorded = []
        for step in steps:
            if step[0] == "update":
                _, drone_id, cell, current_step = step
                manager.update_claimed(drone_id, cell, shared, current_step)
                recorded.append(
                    {
                        "call": "update_claimed",
                        "drone_id": drone_id,
                        "cell": list(cell) if cell is not None else None,
                        "current_step": current_step,
                        "state": snapshot(shared),
                    }
                )
            else:
                _, current_step = step
                manager.flush_messages(shared, current_step)
                recorded.append(
                    {
                        "call": "flush_messages",
                        "drone_id": None,
                        "cell": None,
                        "current_step": current_step,
                        "state": snapshot(shared),
                    }
                )
        out.append({"name": name, "message_latency_steps": latency, "steps": recorded})
    return out


# ----------------------------------------------------------------------
# Formation offsets
# ----------------------------------------------------------------------

# Headings deliberately off-axis and spacings deliberately not powers of two,
# so that both components are non-trivial and the vector-scaling grouping is
# observable. An axis-aligned heading zeroes one component and hides half the
# arithmetic.
FORMATION_CASES: list[tuple[str, str, float, list[str], float]] = [
    ("none_returns_nothing", "none", 80.0, ["a", "b", "c"], 0.7),
    ("single_drone_returns_nothing", "line_abreast", 80.0, ["a"], 0.7),
    ("empty_returns_nothing", "line_abreast", 80.0, [], 0.7),
    ("unrecognized_mode_returns_zeros", "echelon", 80.0, ["a", "b", "c"], 0.7),
    ("line_abreast_pair", "line_abreast", 137.3, ["a", "b"], 0.9128),
    ("line_abreast_axis_aligned", "line_abreast", 100.0, ["a", "b", "c"], 0.0),
    ("line_abreast_negative_heading", "line_abreast", 137.3, ["a", "b", "c"], -2.4011),
    # Nine drones so ranks 3 and 4 appear. Ranks 1 and 2 are powers of two and
    # scale a float exactly, so a formation that stops at five drones cannot
    # tell the two vector-scaling groupings apart however odd its spacing is.
    ("line_abreast_nine", "line_abreast", 137.3, list("abcdefghi"), 0.9128),
    ("v_formation_pair", "v_formation", 137.3, ["a", "b"], 0.9128),
    ("v_formation_axis_aligned", "v_formation", 100.0, ["a", "b", "c"], 0.0),
    ("v_formation_heading_past_pi", "v_formation", 137.3, ["a", "b", "c"], 4.7123),
    ("v_formation_nine", "v_formation", 137.3, list("abcdefghi"), 0.9128),
    ("zero_spacing_is_all_zeros", "line_abreast", 0.0, ["a", "b", "c"], 0.9128),
    ("negative_spacing_mirrors", "line_abreast", -137.3, ["a", "b", "c"], 0.9128),
    # A repeated id: the second assignment overwrites the first in place and
    # the key keeps its original position.
    ("repeated_id_overwrites_in_place", "line_abreast", 137.3, ["a", "b", "a"], 0.9128),
]


def formation_cases() -> list[dict]:
    out = []
    for name, mode, spacing, drone_ids, heading in FORMATION_CASES:
        manager = CoordinationManager(
            CoordinationPolicy(formation_mode=mode, formation_spacing_m=spacing)
        )
        offsets = manager.formation_offsets(list(drone_ids), heading)
        out.append(
            {
                "name": name,
                "formation_mode": mode,
                "formation_spacing_m": exact(spacing),
                "drone_ids": list(drone_ids),
                "lead_heading_rad": exact(heading),
                "offsets": [
                    {"drone_id": did, "offset": [exact(xy[0]), exact(xy[1])]}
                    for did, xy in offsets.items()
                ],
            }
        )
    return out


def _factor_by_factor(mode: str, index: int, spacing: float, heading: float) -> list[float]:
    """The offset a port would get by folding the factors in one at a time.

    This is the plausible wrong spelling: scale the unit vector by `sign`, then
    by `rank`, then by the spacing, rather than forming the whole scalar first.
    It is the same algebra and a different float.
    """
    perp = (-math.sin(heading), math.cos(heading))
    fwd = (math.cos(heading), math.sin(heading))
    sign = 1 if index % 2 == 1 else -1
    rank = (index + 1) // 2
    if mode == "line_abreast":
        return [((perp[k] * sign) * rank) * spacing for k in (0, 1)]
    return [
        -(((fwd[k] * rank) * spacing) * 0.5) + (((perp[k] * sign) * rank) * spacing) * 0.5
        for k in (0, 1)
    ]


def check_discriminating(formations: list[dict]) -> None:
    """Refuse to emit a set that cannot tell the vector-scaling groupings apart.

    The guard is not "are the numbers non-round" -- an earlier version checked
    that and passed while catching nothing. It recomputes each offset the wrong
    way and requires the two to actually disagree somewhere, per formation mode.
    """
    caught: dict[str, int] = {"line_abreast": 0, "v_formation": 0}
    for case in formations:
        mode = case["formation_mode"]
        if mode not in caught:
            continue
        heading = float(case["lead_heading_rad"])
        spacing = float(case["formation_spacing_m"])
        for index, entry in enumerate(case["offsets"]):
            if index == 0:
                continue
            reference = [float(v) for v in entry["offset"]]
            if _factor_by_factor(mode, index, spacing, heading) != reference:
                caught[mode] += 1

    blind = [mode for mode, hits in caught.items() if hits == 0]
    if blind:
        raise SystemExit(
            "no case discriminates the vector-scaling grouping for "
            + ", ".join(sorted(blind))
            + "; ranks 1 and 2 scale exactly, so the formation needs at least "
            "seven drones to reach rank 3"
        )
    print(
        "discriminating offsets: "
        + ", ".join(f"{mode}={hits}" for mode, hits in sorted(caught.items()))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    formations = formation_cases()
    check_discriminating(formations)

    document = {
        "elections": election_cases(),
        "message_sequences": message_cases(),
        "formations": formations,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    print(
        f"wrote {args.out}: {len(document['elections'])} elections, "
        f"{len(document['message_sequences'])} message sequences, "
        f"{len(formations)} formations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
