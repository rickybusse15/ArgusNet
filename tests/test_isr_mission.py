"""Tests for the three-phase ISR mission state machine.

Covers the scanning → localizing → inspecting lifecycle, coverage accumulation,
POI status tracking, and scan_mission_state serialisation.

These tests call run_simulation() directly and require the Rust tracking daemon.
Use short step counts to keep run time low.
"""
from __future__ import annotations

import json
import unittest

from argusnet.core.types import POIStatus, ScanMissionState, to_jsonable
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_scan_result(
    steps: int = 6,
    dt_s: float = 0.5,
    seed: int = 7,
    drone_count: int = 2,
    threshold: float = 0.70,
    poi_count: int = 3,
):
    opts = ScenarioOptions(
        mission_mode="scan_map_inspect",
        drone_count=drone_count,
        scan_coverage_threshold=threshold,
        poi_count=poi_count,
        target_count=0,
    )
    scenario = build_default_scenario(options=opts, seed=seed)
    return run_simulation(scenario, SimulationConfig(steps=steps, dt_s=dt_s, seed=seed))


def _scan_frames(result):
    """Return all frames that have a scan_mission_state."""
    return [f for f in result.frames if f.scan_mission_state is not None]


# ---------------------------------------------------------------------------
# Class 1: Initial state
# ---------------------------------------------------------------------------

class ISRMissionInitialStateTest(unittest.TestCase):

    def test_initial_phase_is_scanning(self) -> None:
        result = _make_scan_result(steps=3)
        frames = _scan_frames(result)
        self.assertGreater(len(frames), 0, "Expected at least one frame with scan_mission_state")
        self.assertEqual(frames[0].scan_mission_state.phase, "scanning")

    def test_initial_coverage_fraction_is_below_complete(self) -> None:
        # Step 0 starts at t=0 with drones at starting positions, so some cells
        # are already marked as scanned.  The key property is that coverage < 100%.
        result = _make_scan_result(steps=2)
        frames = _scan_frames(result)
        self.assertGreater(len(frames), 0)
        self.assertLess(frames[0].scan_mission_state.scan_coverage_fraction, 1.0)

    def test_threshold_preserved_in_state(self) -> None:
        result = _make_scan_result(steps=4, threshold=0.40)
        for frame in _scan_frames(result):
            self.assertAlmostEqual(
                frame.scan_mission_state.scan_coverage_threshold, 0.40, places=6
            )

    def test_scan_mission_state_absent_in_target_tracking_mode(self) -> None:
        opts = ScenarioOptions(
            mission_mode="target_tracking",
            drone_count=2,
            target_count=2,
        )
        scenario = build_default_scenario(options=opts, seed=7)
        result = run_simulation(scenario, SimulationConfig(steps=5, dt_s=0.5, seed=7))
        for frame in result.frames:
            self.assertIsNone(
                frame.scan_mission_state,
                "target_tracking mode must never produce scan_mission_state",
            )


# ---------------------------------------------------------------------------
# Class 2: Coverage accumulation
# ---------------------------------------------------------------------------

class ISRMissionCoverageTest(unittest.TestCase):

    def test_coverage_fraction_increases_monotonically(self) -> None:
        result = _make_scan_result(steps=20, dt_s=0.5)
        frames = _scan_frames(result)
        fracs = [f.scan_mission_state.scan_coverage_fraction for f in frames]
        for prev, curr in zip(fracs, fracs[1:]):
            self.assertGreaterEqual(
                curr, prev - 1e-9,  # allow tiny float rounding
                f"Coverage decreased: {prev} → {curr}",
            )

    def test_coverage_fraction_bounded_0_to_1(self) -> None:
        result = _make_scan_result(steps=20)
        for frame in _scan_frames(result):
            frac = frame.scan_mission_state.scan_coverage_fraction
            self.assertGreaterEqual(frac, 0.0)
            self.assertLessEqual(frac, 1.0)

    def test_newly_scanned_cells_are_coordinate_tuples(self) -> None:
        result = _make_scan_result(steps=20)
        found_cells = False
        for frame in _scan_frames(result):
            cells = frame.scan_mission_state.newly_scanned_cells
            if cells:
                found_cells = True
                for cell in cells:
                    self.assertEqual(len(cell), 3, "Each cell should be a 3-tuple")
                    for v in cell:
                        self.assertIsInstance(v, float)
                break
        self.assertTrue(found_cells, "Expected at least one frame with newly_scanned_cells")

    def test_more_drones_cover_more_area(self) -> None:
        result_1 = _make_scan_result(steps=15, drone_count=1)
        result_4 = _make_scan_result(steps=15, drone_count=4)
        frames_1 = _scan_frames(result_1)
        frames_4 = _scan_frames(result_4)
        cov_1 = frames_1[-1].scan_mission_state.scan_coverage_fraction if frames_1 else 0.0
        cov_4 = frames_4[-1].scan_mission_state.scan_coverage_fraction if frames_4 else 0.0
        self.assertGreaterEqual(
            cov_4, cov_1,
            f"4-drone coverage ({cov_4:.4f}) should be >= 1-drone coverage ({cov_1:.4f})",
        )

    def test_coverage_never_decreases_across_steps(self) -> None:
        result = _make_scan_result(steps=20)
        frames = _scan_frames(result)
        fracs = [f.scan_mission_state.scan_coverage_fraction for f in frames]
        for i in range(1, len(fracs)):
            self.assertGreaterEqual(
                fracs[i], fracs[i - 1] - 1e-9,
                f"Coverage went backward at step {i}: {fracs[i - 1]} → {fracs[i]}",
            )


# ---------------------------------------------------------------------------
# Class 3: Phase transitions (use low threshold to trigger quickly)
# ---------------------------------------------------------------------------

class ISRMissionPhaseTransitionTest(unittest.TestCase):
    """Use scan_coverage_threshold=0.01 so transitions happen in few steps."""

    _THRESHOLD = 0.01

    def _fast_result(self, steps: int = 25):
        return _make_scan_result(steps=steps, threshold=self._THRESHOLD)

    def test_phase_is_valid_string(self) -> None:
        result = _make_scan_result(steps=8)
        valid = {"scanning", "localizing", "inspecting", "complete"}
        for frame in _scan_frames(result):
            self.assertIn(frame.scan_mission_state.phase, valid)

    def test_phase_transitions_to_localizing(self) -> None:
        result = self._fast_result(steps=25)
        phases = [f.scan_mission_state.phase for f in _scan_frames(result)]
        self.assertIn(
            "localizing", phases,
            "Expected at least one 'localizing' frame with threshold=0.01 in 25 steps",
        )

    def test_localizing_comes_after_scanning(self) -> None:
        result = self._fast_result(steps=25)
        seen_localizing = False
        for frame in _scan_frames(result):
            phase = frame.scan_mission_state.phase
            if seen_localizing:
                self.assertNotEqual(
                    phase, "scanning",
                    "Phase reverted to 'scanning' after reaching 'localizing'",
                )
            if phase == "localizing":
                seen_localizing = True

    def test_phase_started_at_s_records_transition_timestamp(self) -> None:
        # phase_started_at_s is set to the simulation timestamp when the phase
        # changes.  With a very low threshold the transition may happen at step 0
        # (t=0.0), so we only verify the value is non-negative and that the
        # phase_started_at_s for a later phase is >= the earlier one.
        result = self._fast_result(steps=25)
        frames = _scan_frames(result)
        for frame in frames:
            self.assertGreaterEqual(
                frame.scan_mission_state.phase_started_at_s, 0.0,
                "phase_started_at_s must always be non-negative",
            )
        # If both scanning and localizing frames exist, the localizing
        # phase_started_at_s must be >= scanning phase_started_at_s.
        scanning_frames = [f for f in frames if f.scan_mission_state.phase == "scanning"]
        localizing_frames = [f for f in frames if f.scan_mission_state.phase == "localizing"]
        if scanning_frames and localizing_frames:
            self.assertGreaterEqual(
                localizing_frames[0].scan_mission_state.phase_started_at_s,
                scanning_frames[0].scan_mission_state.phase_started_at_s,
            )

    def test_localization_estimates_field_is_a_list(self) -> None:
        # localization_estimates is populated progressively as the localizer
        # converges.  We verify the field type is always a list (may be empty
        # in early localizing steps before the grid localizer emits estimates).
        result = self._fast_result(steps=25)
        localizing = [
            f for f in _scan_frames(result)
            if f.scan_mission_state.phase == "localizing"
        ]
        if not localizing:
            self.skipTest("No localizing frames reached in 25 steps")
        for frame in localizing:
            self.assertIsInstance(
                frame.scan_mission_state.localization_estimates, list,
                "localization_estimates must be a list",
            )

    def test_localization_estimate_confidence_in_range(self) -> None:
        result = self._fast_result(steps=25)
        for frame in _scan_frames(result):
            if frame.scan_mission_state.phase != "localizing":
                continue
            for est in frame.scan_mission_state.localization_estimates:
                self.assertGreaterEqual(est.confidence, 0.0)
                self.assertLessEqual(est.confidence, 1.0)


# ---------------------------------------------------------------------------
# Class 4: POI status
# ---------------------------------------------------------------------------

class ISRMissionPOITest(unittest.TestCase):

    def test_poi_count_matches_configured_poi_count(self) -> None:
        result = _make_scan_result(steps=6, poi_count=2)
        for frame in _scan_frames(result):
            ms = frame.scan_mission_state
            self.assertEqual(ms.total_poi_count, 2)
            self.assertEqual(len(ms.poi_statuses), 2)

    def test_initial_poi_statuses_are_pending(self) -> None:
        result = _make_scan_result(steps=4)
        scanning = [
            f for f in _scan_frames(result)
            if f.scan_mission_state.phase == "scanning"
        ]
        self.assertGreater(len(scanning), 0)
        frame = scanning[0]
        for poi in frame.scan_mission_state.poi_statuses:
            self.assertEqual(
                poi.status, "pending",
                f"POI {poi.poi_id} should be 'pending' in scanning phase, got '{poi.status}'",
            )

    def test_poi_status_values_are_valid_strings(self) -> None:
        result = _make_scan_result(steps=10)
        valid = {"pending", "active", "complete"}
        for frame in _scan_frames(result):
            for poi in frame.scan_mission_state.poi_statuses:
                self.assertIn(
                    poi.status, valid,
                    f"Invalid POI status: '{poi.status}'",
                )

    def test_completed_poi_count_never_exceeds_total(self) -> None:
        result = _make_scan_result(steps=10)
        for frame in _scan_frames(result):
            ms = frame.scan_mission_state
            self.assertLessEqual(
                ms.completed_poi_count, ms.total_poi_count,
                f"completed ({ms.completed_poi_count}) > total ({ms.total_poi_count})",
            )

    def test_scan_mission_state_is_json_serializable(self) -> None:
        result = _make_scan_result(steps=3)
        for frame in result.frames:
            serialized = json.dumps(to_jsonable(frame))
            parsed = json.loads(serialized)
            if frame.scan_mission_state is not None:
                self.assertIn("scan_mission_state", parsed)
                sms = parsed["scan_mission_state"]
                self.assertIn("phase", sms)
                self.assertIn("scan_coverage_fraction", sms)
                self.assertIn("poi_statuses", sms)


if __name__ == "__main__":
    unittest.main()
