from __future__ import annotations

import unittest

from argusnet.evaluation.scenarios import get_scenario, list_scenarios
from argusnet.evaluation.scorecard import scorecard_from_replay


class ScorecardAndPresetTest(unittest.TestCase):
    def test_scorecard_from_minimal_replay(self) -> None:
        replay_doc = {
            "meta": {
                "scenario_name": "unit-scorecard",
                "seed": 7,
                "dt_s": 1.0,
                "frame_count": 1,
            },
            "frames": [
                {
                    "timestamp_s": 0.0,
                    "nodes": [],
                    "observations": [],
                    "tracks": [],
                    "truths": [],
                    "metrics": {},
                }
            ],
        }

        scorecard = scorecard_from_replay(
            replay_doc,
            mission_type="mapping",
            difficulty=0.1,
        ).to_dict()

        self.assertEqual(scorecard["scenario_name"], "unit-scorecard")
        self.assertIn("frame_time_p95_ms", scorecard["performance"])
        self.assertIn("comms_dropout_duration_s", scorecard["reliability"])

    def test_scorecard_reports_the_assignment_history(self) -> None:
        def frame(t: float) -> dict:
            keys = ("nodes", "observations", "tracks", "truths")
            return {"timestamp_s": t, **{key: [] for key in keys}, "metrics": {}}

        def event(t: float, kind: str, task: str, **extra: str) -> dict:
            return {"timestamp_s": t, "kind": kind, "task": task, "reason": "r", **extra}

        replay_doc = {
            "meta": {
                "scenario_name": "unit-assignments",
                "seed": 7,
                "dt_s": 1.0,
                "frame_count": 2,
                "assignment_history": {
                    "drone-a": [
                        event(0.0, "assign", "search"),
                        event(5.0, "reassign", "poi:p1"),
                        event(9.0, "handoff", "poi:p1", to="drone-b"),
                        event(20.0, "reassign", "search"),
                    ]
                },
            },
            "frames": [frame(0.0), frame(1.0)],
        }

        mission = scorecard_from_replay(replay_doc, mission_type="inspection").to_dict()["mission"]

        self.assertEqual(mission["task_changes"], 2)
        self.assertEqual(mission["poi_handoffs"], 1)
        self.assertEqual(mission["assignment_thrash"], 1)
        self.assertIn("distance_per_drone_m", mission)
        self.assertIn("battery_used_per_drone", mission)

    def test_operational_presets_are_registered(self) -> None:
        expected = {
            "mapping",
            "inspection",
            "target_tracking",
            "loss_of_signal",
            "large_map",
            "stress",
        }

        self.assertTrue(expected.issubset(set(list_scenarios())))
        self.assertEqual(get_scenario("loss_of_signal").mission_type, "resilience")


if __name__ == "__main__":
    unittest.main()
