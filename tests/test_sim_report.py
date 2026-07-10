"""Tests for the mode-aware simulation report lines.

A run with no fused tracks is not necessarily a failure: in scan_map_inspect
mode there are no live targets by design, and in target_tracking mode the
targets may simply be out of sensor range. These tests pin the human-readable
summary produced for each case.
"""

from __future__ import annotations

import unittest

from argusnet.core.types import ScanMissionState
from argusnet.simulation.sim import (
    REJECT_OUT_OF_RANGE,
    _no_tracks_diagnostic_lines,
    _scan_mission_report_lines,
)


def _scan_state(**overrides) -> ScanMissionState:
    defaults = dict(
        phase="scanning",
        scan_coverage_fraction=0.25,
        scan_coverage_threshold=0.7,
        localization_estimates=[],
        poi_statuses=[],
        completed_poi_count=1,
        total_poi_count=3,
    )
    defaults.update(overrides)
    return ScanMissionState(**defaults)


class ScanMissionReportTests(unittest.TestCase):
    def test_reports_coverage_and_pois(self) -> None:
        lines = _scan_mission_report_lines(_scan_state())
        text = "\n".join(lines)
        self.assertIn("scanning", text)
        self.assertIn("25.0%", text)  # coverage fraction
        self.assertIn("70%", text)  # threshold
        self.assertIn("1/3", text)  # POIs inspected
        self.assertNotIn("No fused tracks", text)

    def test_timeout_note_only_when_timed_out(self) -> None:
        without = "\n".join(_scan_mission_report_lines(_scan_state()))
        self.assertNotIn("timeout", without)
        with_timeout = "\n".join(
            _scan_mission_report_lines(_scan_state(localization_timed_out=True))
        )
        self.assertIn("timeout", with_timeout)


class NoTracksDiagnosticTests(unittest.TestCase):
    def test_no_observations_attempted(self) -> None:
        lines = _no_tracks_diagnostic_lines({"generation_attempted_count": 0})
        text = "\n".join(lines)
        self.assertIn("no observations were attempted", text)
        self.assertIn("--target-count", text)

    def test_out_of_range_hint(self) -> None:
        summary = {
            "generation_attempted_count": 3267,
            "generation_accepted_count": 84,
            "generation_rejection_counts": {REJECT_OUT_OF_RANGE: 3000, "outside_fov": 120},
        }
        text = "\n".join(_no_tracks_diagnostic_lines(summary))
        self.assertIn("84/3267", text)
        self.assertIn(REJECT_OUT_OF_RANGE, text)
        self.assertIn("--map-preset", text)  # actionable hint

    def test_non_range_rejection_has_no_range_hint(self) -> None:
        summary = {
            "generation_attempted_count": 100,
            "generation_accepted_count": 5,
            "generation_rejection_counts": {"outside_fov": 95},
        }
        text = "\n".join(_no_tracks_diagnostic_lines(summary))
        self.assertIn("outside_fov", text)
        self.assertNotIn("--map-preset", text)


if __name__ == "__main__":
    unittest.main()
