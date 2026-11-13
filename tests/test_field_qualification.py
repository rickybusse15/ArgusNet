"""Scoring a replay against a run profile's frozen thresholds."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from argusnet.evaluation.field_qualification import (
    load_thresholds,
    markdown,
    measure,
    score_replay,
    summarize,
)

THRESHOLDS = Path(__file__).resolve().parent.parent / "docs/contracts/field_v1.thresholds.json"


def node(node_id: str, position: list[float], mobile: bool = True) -> dict:
    return {"node_id": node_id, "position": position, "is_mobile": mobile}


def frame(t: float, phase: str, drone: list[float], **extra) -> dict:
    return {
        "timestamp_s": t,
        "nodes": [node("ground", [0.0, 0.0, 0.0], mobile=False), node("drone", drone)],
        "scan_mission_state": {
            "phase": phase,
            "poi_statuses": [{"poi_id": "poi-00", "status": "complete"}],
        },
        "mapping_state": {
            "belief": {
                "known_fraction": 0.96,
                "ground_rmse_m": 0.7,
                "structure_recall": 0.95,
                "structure_precision": 0.97,
                "quality": {"met_fraction": 0.92},
            }
        },
        "energy_reserves": {"drone": {"remaining_wh": 300.0, "reserve_wh": 100.0}},
        **extra,
    }


def passing_replay() -> dict:
    """A drone that leaves its station, flies out past a power line and back."""
    home = [0.0, 0.0, 60.0]
    frames = [frame(0.0, "scanning", home)]
    # Out and back along x at 60 m, 8 m a tick, well over a line at 30 m.
    xs = [8.0 * k for k in range(1, 11)] + [8.0 * k for k in range(9, -1, -1)]
    for k, x in enumerate(xs, start=1):
        phase = "scanning" if k <= 10 else "egress"
        frames.append(frame(0.25 * k, phase, [x, 0.0, 60.0]))
    frames.append(frame(0.25 * (len(xs) + 1), "complete", [0.0, 0.0, 52.0]))
    frames.append(frame(0.25 * (len(xs) + 2), "complete", [0.0, 0.0, 44.0]))
    return {
        "meta": {
            "run_profile": {"id": "field_v1"},
            "power_lines": [
                {
                    "id": "line",
                    "clearance_m": 10.0,
                    "conductor_offsets_m": [-2.0, 2.0],
                    "sag_ratio": 0.03,
                    "supports": [
                        {"xy_m": [40.0, -100.0], "base_z_m": 0.0, "height_m": 30.0},
                        {"xy_m": [40.0, 100.0], "base_z_m": 0.0, "height_m": 30.0},
                    ],
                }
            ],
        },
        "frames": frames,
    }


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return load_thresholds(THRESHOLDS)


def test_a_clean_run_passes_every_gate(thresholds):
    scored = score_replay(passing_replay(), thresholds, repeat_identical=True)
    assert scored["failed"] == []
    assert scored["passed"]
    measured = scored["measured"]
    assert measured["mission_complete"] is True
    assert measured["drones_home"] is True
    # 60 m up, over conductors that hang 6 m below their 30 m supports at
    # mid-span and 2 m either side of the line.
    assert measured["wire_clearance_min_m"] == pytest.approx(36.06, abs=0.05)
    assert measured["energy_reserve_margin_min_wh"] == 200.0
    assert measured["max_step_m"] == pytest.approx(8.0)
    assert set(scored["gates"]) == set(thresholds["per_run"])


@pytest.mark.parametrize(
    ("change", "gate"),
    [
        (lambda r: r["frames"][5].update(world_contacts=[{"kind": "structure"}]), "world_contacts"),
        (
            lambda r: [f["scan_mission_state"].update(phase="egress") for f in r["frames"][-2:]],
            "mission_complete",
        ),
        (
            lambda r: r["frames"][-1]["scan_mission_state"]["poi_statuses"][0].update(
                status="pending"
            ),
            "pois_complete_fraction",
        ),
        (
            lambda r: r["frames"][-1]["nodes"][1].update(position=[40.0, 0.0, 44.0]),
            "drones_home",
        ),
        (
            lambda r: r["frames"][-1]["mapping_state"]["belief"].update(ground_rmse_m=1.4),
            "ground_rmse_m",
        ),
        (
            lambda r: r["frames"][-1]["mapping_state"]["belief"]["quality"].update(
                met_fraction=0.7
            ),
            "quality_met_fraction",
        ),
        (
            lambda r: r["frames"][3]["energy_reserves"]["drone"].update(remaining_wh=90.0),
            "energy_reserve_margin_min_wh",
        ),
        # Through the conductors' height right over the line.
        (
            lambda r: r["frames"][5]["nodes"][1].update(position=[40.0, 0.0, 33.0]),
            "wire_clearance_min_m",
        ),
        (lambda r: r["meta"].update(run_profile={"id": "something_else"}), "run_profile"),
    ],
)
def test_each_gate_fails_on_what_it_guards(thresholds, change, gate):
    replay = passing_replay()
    change(replay)
    scored = score_replay(replay, thresholds, repeat_identical=True)
    assert gate in scored["failed"]
    assert not scored["passed"]


def test_two_drones_too_close_fail_the_separation_gate(thresholds):
    def with_second(apart_m: float) -> dict:
        replay = passing_replay()
        for item in replay["frames"]:
            first = item["nodes"][1]["position"]
            item["nodes"].append(node("other", [first[0], first[1] + apart_m, first[2]]))
        return replay

    far = score_replay(with_second(40.0), thresholds, repeat_identical=True)
    assert far["measured"]["drone_separation_min_m"] == pytest.approx(40.0)
    assert far["passed"], far["failed"]
    near = score_replay(with_second(9.0), thresholds, repeat_identical=True)
    assert near["failed"] == ["drone_separation_min_m"]
    # One drone has nothing to keep apart from.
    alone = score_replay(passing_replay(), thresholds, repeat_identical=True)
    assert alone["gates"]["drone_separation_min_m"]["status"] == "not_applicable"


def test_a_jump_fails_the_step_gate(thresholds):
    replay = passing_replay()
    replay["frames"][5]["nodes"][1]["position"] = [400.0, 0.0, 60.0]
    assert "max_step_m" in score_replay(replay, thresholds, repeat_identical=True)["failed"]


def test_an_unmeasured_requirement_cannot_pass(thresholds):
    replay = passing_replay()
    for item in replay["frames"]:
        del item["mapping_state"]["belief"]["structure_recall"]
    scored = score_replay(replay, thresholds, repeat_identical=True)
    assert scored["gates"]["structure_recall"] == {"status": "fail", "value": None, "min": 0.9}
    # Nor one the caller did not measure.
    assert "repeat_identical" in score_replay(passing_replay(), thresholds)["failed"]
    assert (
        "repeat_identical"
        in (score_replay(passing_replay(), thresholds, repeat_identical=False)["failed"])
    )


def test_wire_clearance_does_not_apply_without_power_lines(thresholds):
    replay = passing_replay()
    replay["meta"]["power_lines"] = []
    scored = score_replay(replay, thresholds, repeat_identical=True)
    assert scored["gates"]["wire_clearance_min_m"]["status"] == "not_applicable"
    assert scored["passed"]


def test_an_empty_replay_measures_nothing_and_fails(thresholds):
    assert measure({"frames": []}) == {}
    assert not score_replay({"frames": []}, thresholds, repeat_identical=True)["passed"]


def test_the_campaign_needs_every_counted_seed_and_enough_of_them(thresholds):
    good = score_replay(passing_replay(), thresholds, repeat_identical=True)
    bad_replay = passing_replay()
    bad_replay["frames"][5]["world_contacts"] = [{"kind": "structure"}]
    bad = score_replay(bad_replay, thresholds, repeat_identical=True)

    def runs(count: int, start: int, role: str, scored: dict) -> list[dict]:
        return [{"seed": start + k, "role": role, **copy.deepcopy(scored)} for k in range(count)]

    heldout = set(range(900, 905))
    full = runs(10, 100, "qualification", good) + runs(5, 900, "heldout", good)
    summary = summarize(full, thresholds, heldout_seeds=heldout)
    assert summary["qualified"], summary["reasons"]
    assert summary["runs"] == 15 and summary["passed"] == 15

    # A failing tuning seed is reported, not counted.
    with_tuning = full + runs(1, 7, "tuning", bad)
    assert summarize(with_tuning, thresholds, heldout_seeds=heldout)["qualified"]

    # One counted failure is enough.
    one_bad = full[:-1] + runs(1, 904, "heldout", bad)
    summary = summarize(one_bad, thresholds, heldout_seeds=heldout)
    assert not summary["qualified"]
    assert summary["failures_by_gate"] == {"world_contacts": [904]}

    # Too few seeds cannot qualify, however well they did.
    few = runs(3, 100, "qualification", good) + runs(5, 900, "heldout", good)
    summary = summarize(few, thresholds, heldout_seeds=heldout)
    assert not summary["qualified"]
    assert "3 qualification seeds" in summary["reasons"][0]


def test_the_report_states_the_verdict_and_its_limits(thresholds):
    good = score_replay(passing_replay(), thresholds, repeat_identical=True)
    runs = [{"seed": seed, "role": "qualification", **good} for seed in range(3)]
    report = {
        "profile": "field_v1",
        "site": "somewhere",
        "world_id": "world",
        "commit": "abc",
        "binary_sha256": "0" * 64,
        "duration_s": 600.0,
        "thresholds": thresholds,
        "summary": summarize(runs, thresholds, heldout_seeds=set()),
        "limits": ["A simulated qualification."],
    }
    text = markdown(report)
    assert "**Not qualified**" in text
    assert "3 qualification seeds, 10 required" in text
    assert "`ground_rmse_m` | max 1.0" in text
    assert "A simulated qualification." in text


def test_thresholds_are_checked_on_load(tmp_path):
    document = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    for change in (
        {"format": "other"},
        {"version": 2},
        {"per_run": {"known_fraction": {"why": "no bound"}}},
    ):
        path = tmp_path / "thresholds.json"
        path.write_text(json.dumps({**document, **change}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_thresholds(path)
