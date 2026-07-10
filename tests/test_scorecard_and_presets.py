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
