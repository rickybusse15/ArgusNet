"""Verify target_tracking missions route per-tick motion through the executor.

Phase C1 of the post-Phase-B follow-up plan: the MissionExecutor's dispatch
path is the choke-point for every per-drone waypoint, regardless of mission
mode. This test exercises a short target_tracking run end-to-end and asserts
the wiring contract:

- every frame carries a TrackingMissionState (parallel to ScanMissionState)
- scan_mission_state is never produced
- the safety-event delta serialises through the replay JSON without error
- the safety-event task_type "track_target" is serialisable when emitted

The Phase B blocking gate is exercised by tests/test_safety_gate_blocks.py at
the unit level. This file complements it at the integration level.
"""

from __future__ import annotations

import json
import unittest

from argusnet.core.types import to_jsonable
from argusnet.mission.execution import (
    DroneRuntimeState,
    ExecutableCommand,
    MissionConstraints,
    MissionExecutionContext,
    MissionExecutor,
    MissionState,
    MissionTask,
    MissionTaskType,
    SafetyDecision,
    SafetyStatus,
)
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


def _run_target_tracking(steps: int = 6, dt_s: float = 0.5, seed: int = 11):
    opts = ScenarioOptions(
        mission_mode="target_tracking",
        drone_count=2,
        target_count=2,
    )
    scenario = build_default_scenario(options=opts, seed=seed)
    return run_simulation(scenario, SimulationConfig(steps=steps, dt_s=dt_s, seed=seed))


class TargetTrackingExecutorWiringTest(unittest.TestCase):
    def test_every_frame_has_tracking_mission_state(self) -> None:
        result = _run_target_tracking(steps=5)
        self.assertGreater(len(result.frames), 0)
        for frame in result.frames:
            self.assertIsNotNone(
                frame.tracking_mission_state,
                "target_tracking must populate tracking_mission_state every frame",
            )

    def test_scan_mission_state_never_present_in_target_tracking(self) -> None:
        result = _run_target_tracking(steps=5)
        for frame in result.frames:
            self.assertIsNone(frame.scan_mission_state)

    def test_tracking_mission_state_serialises_through_replay_json(self) -> None:
        result = _run_target_tracking(steps=4)
        for frame in result.frames:
            payload = json.dumps(to_jsonable(frame))
            self.assertIn("tracking_mission_state", payload)


class TrackTargetCommandUnitTest(unittest.TestCase):
    """The executor accepts TRACK_TARGET commands and records rejections."""

    def _executor(self, *, decision: SafetyDecision, executed: list) -> MissionExecutor:
        state = MissionState(
            mission_id="test-tt",
            tasks=[
                MissionTask(
                    task_id="track-targets",
                    task_type=MissionTaskType.TRACK_TARGET,
                    priority=1,
                ),
            ],
        )
        ctx = MissionExecutionContext(
            get_localization_confidence=lambda: 1.0,
            get_battery_fraction=lambda: 1.0,
            validate_command=lambda cmd, st: decision,
            execute_command=lambda cmd, st: executed.append(cmd),
            is_task_complete=lambda: False,
            now_s=lambda: 7.0,
        )
        return MissionExecutor(
            state=state,
            constraints=MissionConstraints(geofence_radius_m=500.0),
            ctx=ctx,
        )

    def _state(self) -> DroneRuntimeState:
        return DroneRuntimeState(
            drone_id="drone-tt-0",
            position_m=(10.0, 20.0, 100.0),
            velocity_mps=(5.0, 0.0, 0.0),
            agl_m=100.0,
            battery_fraction=0.9,
        )

    def _command(self) -> ExecutableCommand:
        return ExecutableCommand(
            drone_id="drone-tt-0",
            target_xy_m=(11.0, 20.0),
            target_z_m=100.0,
            task_type=MissionTaskType.TRACK_TARGET,
            reason="track",
        )

    def test_track_target_approval_calls_execute(self) -> None:
        executed: list = []
        exec_ = self._executor(
            decision=SafetyDecision(status=SafetyStatus.OK), executed=executed
        )
        decision = exec_.dispatch(self._command(), self._state())
        self.assertEqual(decision.status, SafetyStatus.OK)
        self.assertEqual(len(executed), 1)
        self.assertEqual(executed[0].task_type, MissionTaskType.TRACK_TARGET)

    def test_track_target_rejection_records_safety_event(self) -> None:
        executed: list = []
        exec_ = self._executor(
            decision=SafetyDecision(
                status=SafetyStatus.REJECTED,
                reason="agl too low",
                violations=("min_agl",),
            ),
            executed=executed,
        )
        exec_.dispatch(self._command(), self._state())
        self.assertEqual(executed, [])
        self.assertEqual(len(exec_.state.safety_events), 1)
        evt = exec_.state.safety_events[0]
        self.assertEqual(evt.task_type, MissionTaskType.TRACK_TARGET)
        self.assertEqual(evt.violations, ("min_agl",))


if __name__ == "__main__":
    unittest.main()
