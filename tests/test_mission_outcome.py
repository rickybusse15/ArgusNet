"""F1: every outcome carries reasons that point at the frames behind them."""

from __future__ import annotations

import copy
import unittest

from argusnet.evaluation.outcome import OUTCOME_FORMAT, mission_outcome


def _node(node_id: str, battery: float | None) -> dict:
    return {
        "node_id": node_id,
        "position": [0.0, 0.0, 50.0],
        "velocity": [0.0, 0.0, 0.0],
        "is_mobile": battery is not None,
        "timestamp_s": 0.0,
        "health": 1.0,
        "battery_fraction": battery,
    }


def _frame(tick: int, *, phase: str = "scanning", pois: list[dict] | None = None) -> dict:
    statuses = pois if pois is not None else []
    return {
        "timestamp_s": float(tick),
        "nodes": [_node("drone-a", 1.0 - 0.01 * tick), _node("tower", -1.0)],
        "observations": [],
        "tracks": [],
        "truths": [],
        "metrics": {},
        "scan_mission_state": {
            "phase": phase,
            "poi_statuses": statuses,
            "completed_poi_count": sum(s["status"] == "complete" for s in statuses),
            "total_poi_count": len(statuses),
            "scan_coverage_fraction": 0.8,
            "scan_coverage_threshold": 0.3,
            "localization_timed_out": False,
        },
    }


def _replay(frames: list[dict]) -> dict:
    return {
        "meta": {
            "scenario_name": "unit-outcome",
            "seed": 3,
            "dt_s": 1.0,
            "requested_duration_s": float(len(frames) - 1),
            "frame_count": len(frames),
        },
        "frames": frames,
    }


def _completed() -> dict:
    done = [{"poi_id": "poi-0", "status": "complete"}, {"poi_id": "poi-1", "status": "complete"}]
    frames = [_frame(t) for t in range(9)] + [_frame(9, phase="complete", pois=done)]
    return _replay(frames)


def _codes(outcome: dict) -> list[str]:
    return [reason["code"] for reason in outcome["reasons"]]


class MissionOutcomeTest(unittest.TestCase):
    def test_a_completed_run_says_why(self) -> None:
        outcome = mission_outcome(_completed())
        self.assertEqual(outcome["format"], OUTCOME_FORMAT)
        self.assertEqual(outcome["outcome"], "completed")
        self.assertEqual(_codes(outcome), ["objectives_met"])
        self.assertEqual(outcome["reasons"][0]["evidence"][0]["tick"], 9)
        self.assertEqual(outcome["objectives"]["unmet_pois"], [])

    def test_an_aborted_run_names_the_platform_event_and_tick(self) -> None:
        replay = _completed()
        replay["frames"][4]["safety_events"] = [
            {
                "validation_id": "safety-4-drone-a",
                "subject_id": "drone-a",
                "accepted": False,
                "timestamp_s": 4.0,
                "validator": "blocking_monitor",
                "violations": ["min_agl"],
                "clamped": True,
                "reason": "abort",
            }
        ]
        outcome = mission_outcome(replay)
        self.assertEqual(outcome["outcome"], "aborted")
        abort = outcome["reasons"][0]
        self.assertEqual(abort["code"], "safety_abort")
        self.assertEqual(
            abort["evidence"][0],
            {
                "kind": "safety_event",
                "id": "safety-4-drone-a",
                "tick": 4,
                "violations": ["min_agl"],
            },
        )
        self.assertEqual(outcome["violations"]["by_violation"], {"min_agl": 1})
        self.assertEqual(outcome["violations"]["aborted_platforms"], ["drone-a"])

    def test_a_recovered_run_is_incomplete_whatever_it_achieved(self) -> None:
        replay = _completed()
        replay["meta"]["recovery"] = {"recovered_from_tick": 4}
        outcome = mission_outcome(replay)
        self.assertEqual(outcome["outcome"], "incomplete")
        self.assertEqual(_codes(outcome), ["recovered_after_crash"])

    def test_a_truncated_replay_is_incomplete(self) -> None:
        replay = _completed()
        replay["meta"]["requested_duration_s"] = 20.0
        outcome = mission_outcome(replay)
        self.assertEqual(outcome["outcome"], "incomplete")
        self.assertIn("truncated", _codes(outcome))

    def test_unmet_objectives_list_the_pois_left_and_the_phase(self) -> None:
        replay = _completed()
        pois = [{"poi_id": "poi-0", "status": "complete"}, {"poi_id": "poi-1", "status": "pending"}]
        replay["frames"][-1] = _frame(9, phase="inspecting", pois=pois)
        outcome = mission_outcome(replay)
        self.assertEqual(outcome["outcome"], "objectives_unmet")
        self.assertEqual(_codes(outcome), ["pois_not_inspected", "mission_not_complete"])
        self.assertEqual(outcome["reasons"][0]["evidence"][0]["id"], "poi-1")

    def test_missing_measurements_are_unavailable_not_zero(self) -> None:
        replay = copy.deepcopy(_completed())
        for frame in replay["frames"]:
            for node in frame["nodes"]:
                node["battery_fraction"] = None
        outcome = mission_outcome(replay)
        self.assertEqual(outcome["energy"]["status"], "unavailable")
        sufficiency = outcome["evidence_sufficiency"]
        self.assertEqual(sufficiency["localisation_rmse_m"]["status"], "unavailable")
        self.assertEqual(sufficiency["covariance_reduction_mean"]["status"], "unavailable")

    def test_a_ground_station_sentinel_is_not_an_empty_battery(self) -> None:
        energy = mission_outcome(_completed())["energy"]
        self.assertEqual(energy["min_platform"], "drone-a")
        self.assertNotIn("tower", energy["per_platform"])

    def test_a_late_localization_timeout_is_reported(self) -> None:
        replay = _completed()
        replay["frames"][6]["scan_mission_state"]["localization_timed_out"] = True
        outcome = mission_outcome(replay)
        self.assertEqual(outcome["degradation"]["localization_timed_out_at_tick"], 6)
        self.assertIn("localization_timed_out", _codes(outcome))


if __name__ == "__main__":
    unittest.main()
