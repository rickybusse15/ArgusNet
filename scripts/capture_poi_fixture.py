#!/usr/bin/env python3
"""Record POIManager's lifecycle: assignment, handoff, dwell and completion.

This is a stateful object, so the fixture records SEQUENCES of calls and the
result of each, rather than isolated inputs and outputs. A port that got every
individual method right but kept its POIs in a hash map would still diverge:
insertion order decides which POI is chosen among equals, and which id appears
first in `check_completions`.

Three orderings are load-bearing and all three are pinned by cases below:

* `assign_nearest` sorts by `(-priority, distance)` with `list.sort`, which is
  STABLE -- so two POIs at the same priority and the same distance keep
  construction order.
* `assign_energy_aware` scans in construction order and keeps the FIRST
  strictly-better score.
* `check_completions` iterates in construction order, so the returned ids come
  back in that order rather than in completion order.

Usage:
    python3 scripts/capture_poi_fixture.py --out tests/fixtures/poi.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.core.types import InspectionPOI  # noqa: E402
from argusnet.planning.poi import POIAssignmentContext, POIManager  # noqa: E402


def exact(value: float) -> str:
    return repr(float(value))


def triple(xyz) -> list[str]:
    return [exact(v) for v in xyz]


def poi(pid: str, x: float, y: float, priority: int, dwell: float = 15.0) -> InspectionPOI:
    return InspectionPOI(
        poi_id=pid,
        name=pid.upper(),
        position=np.array([x, y, 0.0], dtype=float),
        priority=priority,
        required_dwell_s=dwell,
    )


def poi_spec(p: InspectionPOI) -> dict:
    return {
        "poi_id": p.poi_id,
        "name": p.name,
        "position": triple(p.position),
        "priority": p.priority,
        "required_dwell_s": exact(p.required_dwell_s),
    }


def statuses_of(manager: POIManager) -> list[dict]:
    return [
        {
            "poi_id": s.poi_id,
            "status": s.status,
            "assigned_drone_id": s.assigned_drone_id,
            "arrival_time_s": None if s.arrival_time_s is None else exact(s.arrival_time_s),
            "completion_time_s": (
                None if s.completion_time_s is None else exact(s.completion_time_s)
            ),
            "dwell_accumulated_s": exact(s.dwell_accumulated_s),
        }
        for s in manager.statuses
    ]


class Recorder:
    """Drives a manager through a call sequence, recording each result."""

    def __init__(self, pois: list[InspectionPOI]) -> None:
        self.pois = pois
        self.manager = POIManager(list(pois))
        self.steps: list[dict] = []

    def _record(self, call: str, args: dict, result) -> None:
        self.steps.append(
            {
                "call": call,
                "args": args,
                "result": result,
                "statuses": statuses_of(self.manager),
                "completed_count": self.manager.completed_count,
                "all_complete": self.manager.all_complete,
            }
        )

    def assign_nearest(self, drone_id: str, pos, battery_fraction: float) -> None:
        chosen = self.manager.assign_nearest(
            drone_id, np.asarray(pos, dtype=float), battery_fraction
        )
        self._record(
            "assign_nearest",
            {
                "drone_id": drone_id,
                "drone_pos": triple(pos),
                "battery_fraction": exact(battery_fraction),
            },
            None if chosen is None else chosen.poi_id,
        )

    def assign_energy_aware(self, **kwargs) -> None:
        context = POIAssignmentContext(
            drone_id=kwargs["drone_id"],
            drone_pos=np.asarray(kwargs["drone_pos"], dtype=float),
            battery_remaining_wh=kwargs["battery_remaining_wh"],
            battery_capacity_wh=kwargs["battery_capacity_wh"],
            cruise_speed_mps=kwargs.get("cruise_speed_mps", 28.0),
            battery_reserve_fraction=kwargs.get("battery_reserve_fraction", 0.20),
        )
        chosen = self.manager.assign_energy_aware(context)
        self._record(
            "assign_energy_aware",
            {
                "drone_id": context.drone_id,
                "drone_pos": triple(context.drone_pos),
                "battery_remaining_wh": exact(context.battery_remaining_wh),
                "battery_capacity_wh": exact(context.battery_capacity_wh),
                "cruise_speed_mps": exact(context.cruise_speed_mps),
                "battery_reserve_fraction": exact(context.battery_reserve_fraction),
            },
            None if chosen is None else chosen.poi_id,
        )

    def accumulate_dwell(self, drone_id: str, pos, dt_s: float) -> None:
        self.manager.accumulate_dwell(drone_id, np.asarray(pos, dtype=float), dt_s)
        self._record(
            "accumulate_dwell",
            {"drone_id": drone_id, "drone_pos": triple(pos), "dt_s": exact(dt_s)},
            None,
        )

    def check_completions(self, timestamp_s: float) -> None:
        done = self.manager.check_completions(timestamp_s)
        self._record("check_completions", {"timestamp_s": exact(timestamp_s)}, list(done))

    def trigger_handoff(self, a: str, b: str) -> None:
        ok = self.manager.trigger_handoff(a, b)
        self._record("trigger_handoff", {"from_drone_id": a, "to_drone_id": b}, bool(ok))

    def request_team_assign(self, poi_id: str, drone_id: str) -> None:
        ok = self.manager.request_team_assign(poi_id, drone_id)
        self._record(
            "request_team_assign", {"poi_id": poi_id, "second_drone_id": drone_id}, bool(ok)
        )

    def rescore(self, densities: dict[str, float]) -> None:
        """Rescore with a stub map returning a per-POI density by id."""

        class _Map:
            def __init__(self, by_id, pois):
                self.by_id = by_id
                self.pois = pois

            def coverage_in_region(self, xy, radius_m):  # noqa: ARG002
                for p in self.pois:
                    if abs(float(p.position[0]) - float(xy[0])) < 1e-9 and (
                        abs(float(p.position[1]) - float(xy[1])) < 1e-9
                    ):
                        return self.by_id[p.poi_id]
                raise KeyError("no POI at that position")

        self.manager.rescore_from_map(_Map(densities, self.pois))
        self._record(
            "rescore_from_map", {"densities": {k: exact(v) for k, v in densities.items()}}, None
        )

    def assignment_for(self, drone_id: str) -> None:
        self._record(
            "assignment_for", {"drone_id": drone_id}, self.manager.assignment_for(drone_id)
        )


def capture() -> dict:
    sequences = []

    # --- priority and distance ordering, and the stable-sort tie -----------
    # d and e share priority AND distance from the origin, so only the stable
    # sort keeping construction order decides between them.
    pois = [
        poi("a", 100.0, 0.0, priority=1),
        poi("b", 10.0, 0.0, priority=1),
        poi("c", 500.0, 0.0, priority=5),
        poi("d", 0.0, 50.0, priority=3),
        poi("e", 50.0, 0.0, priority=3),
    ]
    r = Recorder(pois)
    r.assign_nearest("drone-1", (0.0, 0.0, 0.0), 1.0)  # priority 5 wins despite distance
    r.assign_nearest("drone-1", (0.0, 0.0, 0.0), 1.0)  # already assigned: unchanged
    r.assign_nearest("drone-2", (0.0, 0.0, 0.0), 1.0)  # tie at priority 3: d before e
    r.assign_nearest("drone-3", (0.0, 0.0, 0.0), 0.2)  # below the 25% battery gate
    r.assign_nearest("drone-3", (0.0, 0.0, 0.0), 0.25)  # exactly at the gate: allowed
    r.assign_nearest("drone-4", (0.0, 0.0, 0.0), 1.0)
    r.assign_nearest("drone-5", (0.0, 0.0, 0.0), 1.0)  # nothing left
    sequences.append(
        {"name": "assign_nearest_ordering", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # --- dwell, completion order, and release ------------------------------
    pois = [
        poi("first", 0.0, 0.0, priority=1, dwell=10.0),
        poi("second", 20.0, 0.0, priority=1, dwell=5.0),
    ]
    r = Recorder(pois)
    r.assign_nearest("d1", (0.0, 0.0, 0.0), 1.0)
    r.assign_nearest("d2", (20.0, 0.0, 0.0), 1.0)
    # "second" reaches its dwell first in TIME and is listed SECOND, because
    # check_completions walks the POIs in construction order rather than in
    # the order they finished. Both must complete in the SAME call, or the
    # returned order carries no information and a port keying its POIs by a
    # hash map would pass.
    r.accumulate_dwell("d2", (20.0, 0.0, 0.0), 6.0)
    r.accumulate_dwell("d1", (0.0, 0.0, 0.0), 10.0)
    r.check_completions(100.0)
    r.assignment_for("d1")  # released on completion
    r.assignment_for("d2")
    r.assign_nearest("d3", (0.0, 0.0, 0.0), 1.0)  # nothing pending left
    sequences.append(
        {"name": "dwell_and_completion", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # --- the dwell radius boundary -----------------------------------------
    pois = [poi("p", 0.0, 0.0, priority=1, dwell=1.0)]
    r = Recorder(pois)
    r.assign_nearest("d1", (0.0, 0.0, 0.0), 1.0)
    r.accumulate_dwell("d1", (30.0, 0.0, 0.0), 0.4)  # exactly 30 m: credited
    r.accumulate_dwell("d1", (30.001, 0.0, 0.0), 0.4)  # just outside: not credited
    r.check_completions(10.0)  # 0.4 < 1.0, so not complete
    r.accumulate_dwell("d1", (0.0, 0.0, 0.0), 0.6)
    r.check_completions(20.0)  # 1.0 >= 1.0, complete
    sequences.append(
        {"name": "dwell_radius", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # --- energy-aware assignment -------------------------------------------
    pois = [
        poi("near_low", 100.0, 0.0, priority=1),
        poi("far_high", 5000.0, 0.0, priority=2),
        poi("near_high", 200.0, 0.0, priority=2),
    ]
    r = Recorder(pois)
    # Plenty of energy: priority tiers dominate, energy breaks the tie within
    # a tier, so near_high beats far_high.
    r.assign_energy_aware(
        drone_id="d1",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=100.0,
        battery_capacity_wh=100.0,
    )
    # Reserve eats everything: no assignment at all.
    r.assign_energy_aware(
        drone_id="d2",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=20.0,
        battery_capacity_wh=100.0,
    )
    # Enough to reach the near one only: the 0.8 affordability factor rules
    # the far one out even though the raw energy would cover it.
    r.assign_energy_aware(
        drone_id="d3",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=40.0,
        battery_capacity_wh=100.0,
    )
    r.assign_energy_aware(
        drone_id="d3",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=40.0,
        battery_capacity_wh=100.0,
    )  # already assigned
    sequences.append(
        {"name": "energy_aware", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # --- rescoring, which only the energy-aware path reads ------------------
    # The boost is `round((1 - density) * 3)`, and Python's round() is
    # BANKER'S -- half goes to even. Reaching a tie takes care: the obvious
    # density of 5/6 gives 0.4999999999999999 rather than 0.5, so BOTH rules
    # round it to 0 and the case proves nothing. An earlier version of this
    # fixture did exactly that and passed its own guard, because the winner
    # happened to be the same either way.
    #
    # 0.1666666666666666 -- one 6 short of the nearest double to 1/6 -- makes
    # `(1 - d) * 3` land exactly on 2.5, where the two rules genuinely differ:
    #
    #   banker's:   boost 2 -> far = 3, near = 3  -> a tie, and the NEARER
    #                                                one wins on energy
    #   half-away:  boost 3 -> far = 4, near = 3  -> the far one wins
    #
    # so the choice of rounding rule decides which POI is assigned.
    exact_tie_density = 0.1666666666666666
    if (1.0 - exact_tie_density) * 3.0 != 2.5:
        raise AssertionError(
            "the density no longer lands on a rounding tie; the case cannot "
            "distinguish banker's rounding from rounding half away from zero"
        )
    pois = [
        poi("far_half_boost", 3000.0, 0.0, priority=1),
        poi("near_high_priority", 100.0, 0.0, priority=3),
    ]
    r = Recorder(pois)
    r.rescore({"far_half_boost": exact_tie_density, "near_high_priority": 1.0})
    r.assign_energy_aware(
        drone_id="d1",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=100.0,
        battery_capacity_wh=100.0,
    )
    sequences.append(
        {"name": "rescore_banker_rounding", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # `assign_nearest` reads the RAW priority and ignores rescoring entirely,
    # which `assign_energy_aware` does not. Recorded because it is surprising.
    # Raw priorities that DISAGREE with the boosted ones: `high_raw` wins on
    # the raw priority the reference reads, `boosted` would win if
    # `assign_nearest` consulted the effective priorities the way
    # `assign_energy_aware` does. It does not, and that asymmetry is only
    # visible when the two disagree.
    pois = [
        poi("high_raw", 100.0, 0.0, priority=3),
        poi("boosted", 200.0, 0.0, priority=1),
    ]
    r = Recorder(pois)
    r.rescore({"high_raw": 1.0, "boosted": 0.0})  # +0 -> 3, and +3 -> 4
    r.assign_nearest("d1", (0.0, 0.0, 0.0), 1.0)
    sequences.append(
        {
            "name": "rescore_ignored_by_assign_nearest",
            "pois": [poi_spec(p) for p in pois],
            "steps": r.steps,
        }
    )

    # --- handoff and team assignment ---------------------------------------
    pois = [poi("shared", 0.0, 0.0, priority=1, dwell=10.0), poi("other", 400.0, 0.0, priority=1)]
    r = Recorder(pois)
    r.trigger_handoff("d1", "d2")  # d1 has nothing: False
    r.assign_nearest("d1", (0.0, 0.0, 0.0), 1.0)
    r.trigger_handoff("d1", "d2")  # True
    r.assignment_for("d1")
    r.assignment_for("d2")
    r.request_team_assign("shared", "d3")  # True
    r.request_team_assign("shared", "d3")  # d3 already assigned: False
    r.request_team_assign("other", "d4")  # not active: False
    # Both drones credit the same POI, so dwell accrues twice as fast.
    r.accumulate_dwell("d2", (0.0, 0.0, 0.0), 3.0)
    r.accumulate_dwell("d3", (0.0, 0.0, 0.0), 3.0)
    r.check_completions(50.0)  # 6 < 10
    r.accumulate_dwell("d2", (0.0, 0.0, 0.0), 2.0)
    r.accumulate_dwell("d3", (0.0, 0.0, 0.0), 2.0)
    r.check_completions(60.0)  # 10 >= 10, and BOTH drones are released
    r.assignment_for("d2")
    r.assignment_for("d3")
    sequences.append(
        {"name": "handoff_and_team", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # --- edges the earlier sequences do not reach --------------------------
    #
    # Each of these closes a mutation that survived the first pass.

    # Usable energy of EXACTLY zero, with a POI at EXACTLY the drone's
    # position. Both halves matter: with `< 0.0` instead of `<= 0.0` the guard
    # lets the call through, and a POI ten metres away is then rejected by the
    # affordability gate anyway (its cost exceeds zero), so the two guards look
    # identical. A zero-distance POI costs nothing, passes the gate, and gets
    # assigned -- which only the `<= 0.0` guard prevents.
    pois = [poi("underfoot", 0.0, 0.0, priority=1)]
    r = Recorder(pois)
    r.assign_energy_aware(
        drone_id="d1",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=20.0,
        battery_capacity_wh=100.0,
    )
    # A cruise speed of zero, which the 0.1 m/s floor turns into an enormous
    # but finite cost rather than an infinity. Nothing else in the fixture
    # goes below the floor.
    r2 = Recorder([poi("underfoot", 10.0, 0.0, priority=1)])
    r2.assign_energy_aware(
        drone_id="d1",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=100.0,
        battery_capacity_wh=100.0,
        cruise_speed_mps=0.0,
    )
    r.steps.extend(r2.steps)
    sequences.append(
        {"name": "energy_edges", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # Two POIs with identical effective priority AND identical distance, so
    # their scores are exactly equal and only "the first strictly-better wins"
    # separates them. `>=` would return the later one.
    pois = [
        poi("first_of_two", 0.0, 100.0, priority=2),
        poi("second_of_two", 100.0, 0.0, priority=2),
    ]
    r = Recorder(pois)
    r.assign_energy_aware(
        drone_id="d1",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=100.0,
        battery_capacity_wh=100.0,
    )
    sequences.append(
        {"name": "energy_exact_tie", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # Rescoring that CHANGES the energy-aware winner: raw priorities favour
    # the near POI, boosted ones favour the far one. Without this the
    # effective and raw priorities agree and reading either passes.
    pois = [poi("far_boosted", 3000.0, 0.0, priority=1), poi("near_raw", 100.0, 0.0, priority=3)]
    r = Recorder(pois)
    r.rescore({"far_boosted": 0.0, "near_raw": 1.0})  # +3 -> 4, and +0 -> 3
    r.assign_energy_aware(
        drone_id="d1",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=100.0,
        battery_capacity_wh=100.0,
    )
    sequences.append(
        {
            "name": "rescore_changes_the_winner",
            "pois": [poi_spec(p) for p in pois],
            "steps": r.steps,
        }
    )

    # A POI the map cannot answer for, and one that is already Active when
    # rescoring runs. The reference swallows the lookup failure and assumes a
    # density of 1.0 (no boost), and skips anything not pending.
    pois = [
        poi("active_already", 50.0, 0.0, priority=1),
        poi("unknown_to_the_map", 400.0, 0.0, priority=1),
        poi("known", 500.0, 0.0, priority=1),
    ]
    r = Recorder(pois)
    r.assign_nearest("d1", (0.0, 0.0, 0.0), 1.0)  # makes active_already Active
    r.rescore({"active_already": 0.0, "known": 0.0})  # unknown_to_the_map is missing
    r.assign_energy_aware(
        drone_id="d2",
        drone_pos=(0.0, 0.0, 0.0),
        battery_remaining_wh=100.0,
        battery_capacity_wh=100.0,
    )
    sequences.append(
        {"name": "rescore_edges", "pois": [poi_spec(p) for p in pois], "steps": r.steps}
    )

    # Sized and shaped so an unstable sort actually reorders.
    #
    # Measured, not guessed. Rust's `sort_unstable_by` preserves order for a
    # handful of elements whatever their keys, AND for any number of elements
    # when every key is equal -- pdqsort's all-equal partition never moves
    # anything. It reorders from about forty elements once there are SEVERAL
    # distinct key groups. A five-element tie and a twenty-four-element block
    # of identical keys each failed to tell stable from unstable.
    #
    # So: 48 POIs over four priority tiers and three distances, giving twelve
    # groups of four tied members. The winner is the first-constructed member
    # of the (highest priority, nearest) group -- index 3.
    ring = [
        poi(
            f"ring{index:02d}",
            100.0 * float((index // 4) % 3 + 1),
            0.0,
            priority=(index % 4) + 1,
        )
        for index in range(48)
    ]
    r = Recorder(ring)
    r.assign_nearest("d1", (0.0, 0.0, 0.0), 1.0)
    sequences.append(
        {"name": "stable_sort_at_scale", "pois": [poi_spec(p) for p in ring], "steps": r.steps}
    )

    # --- guards ------------------------------------------------------------
    flat = [{"name": s["name"], "calls": [st["call"] for st in s["steps"]]} for s in sequences]
    results = [(s["name"], st["call"], st["result"]) for s in sequences for st in s["steps"]]
    if not any(call == "trigger_handoff" and res is True for _, call, res in results):
        raise AssertionError("no successful handoff")
    if not any(call == "trigger_handoff" and res is False for _, call, res in results):
        raise AssertionError("no rejected handoff")
    if not any(call == "request_team_assign" and res is True for _, call, res in results):
        raise AssertionError("no successful team assignment")
    if not any(call == "request_team_assign" and res is False for _, call, res in results):
        raise AssertionError("no rejected team assignment")
    if not any(call.startswith("assign_") and res is None for _, call, res in results):
        raise AssertionError("no refused assignment")
    multi = [
        res for _, call, res in results if call == "check_completions" and res and len(res) > 1
    ]
    if not multi:
        raise AssertionError(
            "no check_completions returns more than one id, so the returned "
            "ORDER is unobservable and a port using a hash map would pass"
        )

    return {
        "source": "src/argusnet/planning/poi.py",
        "note": (
            "POIManager driven through call sequences and recorded step by "
            "step. Construction order decides ties in both assignment paths "
            "and the order check_completions returns ids in, so the fixture "
            "contains cases that can only be passed by preserving it."
        ),
        "travel_power_w": exact(350.0),
        "sequences": sequences,
        "call_summary": flat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    steps = sum(len(s["steps"]) for s in document["sequences"])
    print(f"wrote {args.out}: {len(document['sequences'])} sequences, {steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
