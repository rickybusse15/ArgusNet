#!/usr/bin/env python3
"""Record DeconflictionLayer.resolve_step: corridor holds and pairwise yields.

Order decides the outcome three separate times, and the fixture pins each:

* `_yielder` breaks a priority tie on the LEXICOGRAPHICALLY LARGER id -- the
  larger one yields. That is the opposite of the obvious guess, and a port
  that yielded the smaller would pass every case where the roles differ.
* The pairwise pass walks `sorted(adjusted.keys())` and writes back in place,
  so a later pair sees an earlier pair's adjustment. Three drones in a row
  give a different answer than three resolved independently.
* Corridor membership follows the order of the PROPOSED mapping, not sorted
  order, and a `held` set short-circuits later pairs -- so which drone is held
  depends on input order.

Usage:
    python3 scripts/capture_deconfliction_fixture.py --out tests/fixtures/deconfliction.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.core.types import MissionZone  # noqa: E402
from argusnet.planning.deconfliction import (  # noqa: E402
    ROLE_PRIORITY,
    DeconflictionConfig,
    DeconflictionLayer,
)
from argusnet.planning.inspection import FlightCorridor  # noqa: E402


def exact(value: float) -> str:
    return repr(float(value))


def triple(xyz) -> list[str]:
    return [exact(v) for v in xyz]


class FlatTerrain:
    """A terrain whose surface is a constant height, so `clamp_altitude` has a
    single obvious answer and the port can reproduce it without the world
    crate."""

    def __init__(self, height_m: float) -> None:
        self.height_m = height_m

    def clamp_altitude(self, xy, z, min_agl_m):  # noqa: ARG002
        return max(float(z), self.height_m + float(min_agl_m))


def zone(zone_id: str, x: float, y: float, radius: float) -> MissionZone:
    return MissionZone(
        zone_id=zone_id,
        zone_type="exclusion",
        center=np.array([x, y, 0.0], dtype=float),
        radius_m=radius,
    )


def corridor(
    cid: str,
    waypoints: list[list[float]],
    *,
    width_m: float = 50.0,
    direction: str = "bidirectional",
    assigned: list[str] | None = None,
    window: list[float] | None = None,
) -> FlightCorridor:
    return FlightCorridor(
        corridor_id=cid,
        waypoints_xy_m=waypoints,
        width_m=width_m,
        direction=direction,
        assigned_drone_ids=assigned or [],
        active_window=window if window is not None else [0.0, 600.0],
    )


def corridor_spec(c: FlightCorridor) -> dict:
    return {
        "corridor_id": c.corridor_id,
        "waypoints_xy_m": [[exact(v) for v in p] for p in c.waypoints_xy_m],
        "width_m": exact(c.width_m),
        "direction": c.direction,
        "assigned_drone_ids": list(c.assigned_drone_ids),
        "active_window": [exact(v) for v in c.active_window],
    }


def run_case(
    name: str,
    *,
    proposed: list[tuple[str, tuple, tuple]],
    roles: dict[str, str],
    timestamp_s: float = 0.0,
    config: DeconflictionConfig | None = None,
    terrain_height_m: float | None = None,
    zones: list[MissionZone] | None = None,
    corridors: list[FlightCorridor] | None = None,
) -> dict:
    cfg = config or DeconflictionConfig()
    layer = DeconflictionLayer(
        cfg,
        roles,
        terrain=None if terrain_height_m is None else FlatTerrain(terrain_height_m),
        exclusion_zones=zones,
        corridors=corridors,
    )
    # A plain dict preserves insertion order, which corridor membership reads.
    ordered = {
        drone_id: (np.array(pos, dtype=float), np.array(vel, dtype=float))
        for drone_id, pos, vel in proposed
    }
    adjusted, events = layer.resolve_step(ordered, timestamp_s)
    return {
        "name": name,
        "timestamp_s": exact(timestamp_s),
        "config": {
            "min_separation_m": exact(cfg.min_separation_m),
            "look_ahead_s": exact(cfg.look_ahead_s),
            "vertical_separation_m": exact(cfg.vertical_separation_m),
            "min_agl_m": exact(cfg.min_agl_m),
            "corridor_window_default_s": exact(cfg.corridor_window_default_s),
            "enabled": cfg.enabled,
        },
        "terrain_height_m": None if terrain_height_m is None else exact(terrain_height_m),
        "zones": [
            {
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "center": triple(z.center),
                "radius_m": exact(z.radius_m),
            }
            for z in (zones or [])
        ],
        "corridors": [corridor_spec(c) for c in (corridors or [])],
        "roles": dict(roles),
        # Order preserved: corridor membership reads it.
        "proposed": [
            {"drone_id": d, "position": triple(p), "velocity": triple(v)} for d, p, v in proposed
        ],
        "adjusted": [
            {
                "drone_id": d,
                "position": triple(adjusted[d][0]),
                "velocity": triple(adjusted[d][1]),
            }
            for d, _, _ in proposed
        ],
        "events": [
            {
                "yielding_drone_id": e.yielding_drone_id,
                "conflicting_drone_id": e.conflicting_drone_id,
                "predicted_separation_m": exact(e.predicted_separation_m),
                "resolution": e.resolution,
                "timestamp_s": exact(e.timestamp_s),
            }
            for e in events
        ],
    }


def capture() -> dict:
    cases = []

    # --- nothing to do -----------------------------------------------------
    cases.append(
        run_case(
            "far_apart",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (500.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
        )
    )
    cases.append(
        run_case(
            "one_drone_only",
            proposed=[("a", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0))],
            roles={"a": "relay"},
        )
    )
    cases.append(
        run_case(
            "disabled",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (5.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            config=DeconflictionConfig(enabled=False),
        )
    )

    # --- role priority, and the lexicographic tie --------------------------
    cases.append(
        run_case(
            "role_priority_decides",
            proposed=[
                ("alpha", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("bravo", (12.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            # bravo is the lower priority, so bravo yields regardless of id.
            roles={"alpha": "primary_observer", "bravo": "reserve"},
        )
    )
    cases.append(
        run_case(
            "role_priority_reversed",
            proposed=[
                ("alpha", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("bravo", (12.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"alpha": "reserve", "bravo": "primary_observer"},
        )
    )
    cases.append(
        run_case(
            "equal_roles_larger_id_yields",
            proposed=[
                ("alpha", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("bravo", (12.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            # Same role, so the tie-break decides -- and it is the LARGER id
            # that yields, which is the opposite of the obvious guess.
            roles={"alpha": "relay", "bravo": "relay"},
        )
    )
    cases.append(
        run_case(
            "unknown_roles_tie_at_zero",
            proposed=[
                ("alpha", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("bravo", (12.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            # Neither role is in the table, so both score the default of 0.
            roles={"alpha": "nonsense", "bravo": "also_nonsense"},
        )
    )

    # --- the pairwise pass is sequential -----------------------------------
    # Three drones in a line. `b`'s adjustment from the (a, b) pair is visible
    # to the (b, c) pair, so the result is not what resolving each pair
    # independently would give.
    cases.append(
        run_case(
            "three_in_a_row_sequential",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (5.0, 0.0, 0.0)),
                ("b", (10.0, 0.0, 100.0), (5.0, 0.0, 0.0)),
                ("c", (20.0, 0.0, 100.0), (5.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "relay", "c": "reserve"},
        )
    )

    # --- lateral, then the opposite side, then vertical --------------------
    cases.append(
        run_case(
            "lateral_offset",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (0.0, 10.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )
    # An exclusion zone on the first side pushes the yield to the other one.
    #
    # Getting the FLIP to succeed takes deliberate geometry: the first push is
    # already directed away from the peer, so the opposite push heads toward
    # it and is usually still too close, and the layer falls through to a
    # vertical offset instead. Here the two drones are far apart but closing
    # head-on, so the conflict is predicted rather than current -- which
    # leaves room on both sides. An earlier version of this case produced a
    # vertical offset and never reached the flip at all.
    cases.append(
        run_case(
            "lateral_blocked_flips_side",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (0.0, 15.0, 0.0)),
                ("b", (0.0, 40.0, 100.0), (0.0, -15.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
            zones=[zone("z1", 10.0, 40.0, 5.0)],
        )
    )
    # The same geometry with no zone, so the FIRST side is taken -- the pair
    # differs only in whether the flip fires.
    cases.append(
        run_case(
            "lateral_first_side_when_clear",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (0.0, 15.0, 0.0)),
                ("b", (0.0, 40.0, 100.0), (0.0, -15.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )
    # Both sides blocked, so it climbs. `y_pos[2] >= p_pos[2]` decides which
    # way, and the comparison is inclusive.
    cases.append(
        run_case(
            "vertical_offset_up",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (0.0, 10.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
            zones=[zone("z1", 0.0, 37.0, 20.0), zone("z2", 0.0, -17.0, 20.0)],
        )
    )
    cases.append(
        run_case(
            "vertical_offset_down",
            proposed=[
                ("a", (0.0, 0.0, 200.0), (10.0, 0.0, 0.0)),
                ("b", (0.0, 10.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            # b is BELOW a, so b descends.
            roles={"a": "primary_observer", "b": "reserve"},
            zones=[zone("z1", 0.0, 37.0, 20.0), zone("z2", 0.0, -17.0, 20.0)],
        )
    )
    # And terrain clamps the descent back up.
    cases.append(
        run_case(
            "vertical_offset_clamped_by_terrain",
            proposed=[
                ("a", (0.0, 0.0, 200.0), (10.0, 0.0, 0.0)),
                ("b", (0.0, 10.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
            zones=[zone("z1", 0.0, 37.0, 20.0), zone("z2", 0.0, -17.0, 20.0)],
            terrain_height_m=95.0,
        )
    )

    # --- a stationary yielder has no heading to offset from ----------------
    cases.append(
        run_case(
            "stationary_yielder_pushes_away",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (0.0, 0.0, 0.0)),
                ("b", (8.0, 0.0, 100.0), (0.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )
    # Exactly coincident and stationary: no heading AND no separation vector,
    # so the fallback perpendicular of (1, 0) is what remains.
    cases.append(
        run_case(
            "coincident_and_stationary",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (0.0, 0.0, 0.0)),
                ("b", (0.0, 0.0, 100.0), (0.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )

    # --- corridors ---------------------------------------------------------
    line = [[0.0, 0.0], [400.0, 0.0]]
    cases.append(
        run_case(
            "corridor_head_on_hold",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line)],
        )
    )
    # The same pair one window later: the open direction has flipped, so the
    # other drone holds.
    cases.append(
        run_case(
            "corridor_window_flips",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line)],
            timestamp_s=45.0,
        )
    )
    cases.append(
        run_case(
            "corridor_same_direction_ignored",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line)],
        )
    )
    cases.append(
        run_case(
            "corridor_one_way_ignored",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line, direction="inbound")],
        )
    )
    cases.append(
        run_case(
            "corridor_outside_active_window",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line, window=[1000.0, 2000.0])],
            timestamp_s=0.0,
        )
    )
    cases.append(
        run_case(
            "corridor_assignment_filters_members",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
                ("c", (200.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve", "c": "reserve"},
            corridors=[corridor("c1", line, assigned=["a", "b"])],
        )
    )
    cases.append(
        run_case(
            "corridor_too_far_off_centreline",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 400.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line, width_m=20.0)],
        )
    )
    # A bent corridor, so the nearest-segment search matters.
    cases.append(
        run_case(
            "bent_corridor",
            proposed=[
                ("a", (100.0, 10.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (400.0, 90.0, 100.0), (0.0, -10.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0]])],
        )
    )
    # Held drones are removed from the corridor pass by the `held` set, so a
    # third drone cannot hold one that is already holding.
    cases.append(
        run_case(
            "corridor_held_set_short_circuits",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (200.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
                ("c", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve", "c": "corridor_watcher"},
            corridors=[corridor("c1", line)],
        )
    )

    # --- edges the earlier cases do not reach ------------------------------
    #
    # Each closes a mutation that survived the first pass.

    # The pairwise walk sorts ids, so supplying them out of order must not
    # change the result. Every earlier case happened to be already sorted.
    cases.append(
        run_case(
            "proposed_order_is_not_sorted_order",
            proposed=[
                ("zulu", (0.0, 0.0, 100.0), (5.0, 0.0, 0.0)),
                ("alpha", (10.0, 0.0, 100.0), (5.0, 0.0, 0.0)),
                ("mike", (20.0, 0.0, 100.0), (5.0, 0.0, 0.0)),
            ],
            roles={"zulu": "relay", "alpha": "relay", "mike": "relay"},
        )
    )

    # Predicted separation EXACTLY at the minimum. The guard is `>=`, so this
    # is not a conflict; `>` would make it one.
    cases.append(
        run_case(
            "separation_exactly_at_the_minimum",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (0.0, 0.0, 0.0)),
                ("b", (16.0, 0.0, 100.0), (0.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )

    # The flip-away step, which is harder to make observable than it looks.
    #
    # Dropping it is usually absorbed: the un-flipped push heads toward the
    # peer, the still-close check rejects it, and the opposite-side fallback
    # lands on exactly the side the flip would have chosen. It only shows when
    # the toward-the-peer side is ALSO viable, so the two choices genuinely
    # differ rather than converging.
    #
    # Here the pair closes head-on with a small lateral offset, so the
    # perpendicular has a positive component toward the peer (the flip fires)
    # and both sides clear the minimum separation afterwards.
    cases.append(
        run_case(
            "flip_away_when_both_sides_are_viable",
            proposed=[
                ("a", (5.0, 0.0, 100.0), (0.0, 15.0, 0.0)),
                ("b", (0.0, 40.0, 100.0), (0.0, -15.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )
    # The peer on the left of the heading, where the flip fires but the
    # fallback would reach the same place -- kept because it exercises the
    # branch even though it cannot discriminate on its own.
    cases.append(
        run_case(
            "peer_on_the_left_forces_the_flip_away",
            proposed=[
                ("a", (0.0, 10.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (0.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
        )
    )

    # The mirror side lies inside a zone while still being far enough away, so
    # only the zone check keeps it from being taken.
    cases.append(
        run_case(
            "opposite_side_blocked_by_a_zone",
            proposed=[
                ("a", (0.0, 0.0, 100.0), (0.0, 15.0, 0.0)),
                ("b", (0.0, 40.0, 100.0), (0.0, -15.0, 0.0)),
            ],
            roles={"a": "primary_observer", "b": "reserve"},
            zones=[zone("z1", 10.0, 40.0, 5.0), zone("z2", -10.0, 40.0, 5.0)],
        )
    )

    # A corridor with a single-element window is always active.
    cases.append(
        run_case(
            "corridor_with_a_short_window_is_active",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line, window=[0.0])],
        )
    )

    # A platform that would be held TWICE. Without the `held` set the second
    # pair emits another hold for a platform that is already stopped.
    cases.append(
        run_case(
            "held_set_prevents_a_second_hold",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (200.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
                ("c", (300.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve", "c": "relay"},
            corridors=[corridor("c1", line)],
        )
    )

    # Exactly on the corridor's half-width. The comparison is `> width`, so
    # this platform is a member; `>=` would drop it and there would be no pair.
    cases.append(
        run_case(
            "corridor_member_exactly_at_the_width",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 50.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line, width_m=50.0)],
        )
    )

    # A platform inside the corridor but moving straight across it has no
    # along-axis speed, so it is not a member at all.
    cases.append(
        run_case(
            "corridor_crossing_traffic_is_not_a_member",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (0.0, 10.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line)],
        )
    )

    # The window length is derived from corridor length over mean speed --
    # 400 m at 10 m/s is 40 s, against a 5 s default. At t = 50 the two give
    # window indices 1 and 10, odd and even, so the open direction differs and
    # the other platform holds.
    cases.append(
        run_case(
            "corridor_window_from_length_not_the_default",
            proposed=[
                ("a", (100.0, 0.0, 100.0), (10.0, 0.0, 0.0)),
                ("b", (300.0, 0.0, 100.0), (-10.0, 0.0, 0.0)),
            ],
            roles={"a": "relay", "b": "reserve"},
            corridors=[corridor("c1", line)],
            timestamp_s=50.0,
        )
    )

    # --- guards ------------------------------------------------------------
    resolutions = {e["resolution"] for c in cases for e in c["events"]}
    for expected in ("lateral_offset", "vertical_offset", "corridor_hold"):
        if expected not in resolutions:
            raise AssertionError(f"no case produces a {expected!r} resolution")
    if not any(not c["events"] for c in cases):
        raise AssertionError("no case resolves cleanly with no events")
    multi = [c for c in cases if len(c["events"]) > 1]
    if not multi:
        raise AssertionError(
            "no case emits more than one event, so the event ORDER -- which "
            "follows the sorted-id pairwise walk -- is unobservable"
        )
    # The opposite-side flip must actually fire somewhere, and it must land on
    # the mirror of the first side rather than falling through to a vertical
    # offset.
    flipped = next(c for c in cases if c["name"] == "lateral_blocked_flips_side")
    clear = next(c for c in cases if c["name"] == "lateral_first_side_when_clear")
    if not flipped["events"] or flipped["events"][0]["resolution"] != "lateral_offset":
        got = flipped["events"][0]["resolution"] if flipped["events"] else "nothing"
        raise AssertionError(
            f"the blocked-side case resolved as {got!r}; it must stay lateral, "
            "or the opposite-side branch is never reached"
        )
    flipped_y = float(flipped["adjusted"][1]["position"][0])
    clear_y = float(clear["adjusted"][1]["position"][0])
    if flipped_y == clear_y:
        raise AssertionError("blocking the first side changed nothing, so the flip is unobservable")

    # The lexicographic tie-break must actually be exercised, and it must
    # produce the LARGER id as the yielder.
    tie = next(c for c in cases if c["name"] == "equal_roles_larger_id_yields")
    if not tie["events"] or tie["events"][0]["yielding_drone_id"] != "bravo":
        raise AssertionError(
            "the equal-role case did not make the larger id yield; the "
            "tie-break is not being exercised"
        )
    # The derived window must differ from the default in its OPEN DIRECTION,
    # or a port using the default everywhere would pass.
    derived = next(c for c in cases if c["name"] == "corridor_window_from_length_not_the_default")
    base = next(c for c in cases if c["name"] == "corridor_head_on_hold")
    if not derived["events"]:
        raise AssertionError("the derived-window case produced no hold")
    if derived["events"][0]["yielding_drone_id"] == base["events"][0]["yielding_drone_id"]:
        raise AssertionError(
            "the derived window chose the same holder as the first window, so "
            "using the default duration instead would go unnoticed"
        )

    # The flip must land on the side AWAY from the peer, and both sides must
    # be viable so the opposite-side fallback cannot reach the same answer.
    flip = next(c for c in cases if c["name"] == "flip_away_when_both_sides_are_viable")
    if not flip["events"] or flip["events"][0]["resolution"] != "lateral_offset":
        raise AssertionError("the flip case must resolve laterally")
    yielded_x = float(flip["adjusted"][1]["position"][0])
    peer_x = float(flip["proposed"][0]["position"][0])
    if not (yielded_x < 0.0 < peer_x):
        raise AssertionError(
            f"the yielder moved to x={yielded_x} with the peer at x={peer_x}; the "
            "push must go AWAY from the peer, or the flip is not being exercised"
        )

    # The held set must actually short-circuit a second hold.
    twice = next(c for c in cases if c["name"] == "held_set_prevents_a_second_hold")
    if len(twice["events"]) != 1:
        raise AssertionError(
            f"the held-set case emitted {len(twice['events'])} events; it must emit "
            "exactly one, or the set is not short-circuiting anything"
        )

    # The corridor window must genuinely flip who holds.
    first = next(c for c in cases if c["name"] == "corridor_head_on_hold")
    later = next(c for c in cases if c["name"] == "corridor_window_flips")
    held_first = [e for e in first["events"] if e["resolution"] == "corridor_hold"]
    held_later = [e for e in later["events"] if e["resolution"] == "corridor_hold"]
    if not held_first or not held_later:
        raise AssertionError("a corridor case produced no hold")
    if held_first[0]["yielding_drone_id"] == held_later[0]["yielding_drone_id"]:
        raise AssertionError(
            "the same drone holds in both windows, so the alternating direction is unobservable"
        )

    return {
        "source": "src/argusnet/planning/deconfliction.py",
        "note": (
            "DeconflictionLayer.resolve_step recorded per case. Order decides "
            "the outcome three times over: the lexicographic tie-break yields "
            "the LARGER id, the pairwise pass writes back in place so later "
            "pairs see earlier adjustments, and corridor membership follows "
            "the proposed mapping's order rather than sorted order."
        ),
        "role_priority": {role: value for role, value in ROLE_PRIORITY.items()},
        "default_role_priority": 0,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    events = sum(len(c["events"]) for c in document["cases"])
    print(f"wrote {args.out}: {len(document['cases'])} cases, {events} events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
