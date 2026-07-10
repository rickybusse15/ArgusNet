from __future__ import annotations

import unittest

from argusnet.core.types import PlatformHealthState, PlatformLinkState
from argusnet.mission.execution import (
    ExecutableCommand,
    MissionConstraints,
    MissionExecutionContext,
    MissionExecutor,
    MissionState,
    MissionTask,
    MissionTaskStatus,
    MissionTaskType,
    SafetyDecision,
    SafetyStatus,
)
from argusnet.mission.health import PlatformHealthMonitor


class PlatformHealthMonitorTest(unittest.TestCase):
    def test_low_rssi_triggers_lost_link_action(self) -> None:
        monitor = PlatformHealthMonitor(min_rssi_dbm=-85.0)
        health = PlatformHealthState(
            platform_id="drone-1",
            timestamp_s=10.0,
            rssi_dbm=-90.0,
            last_seen_s=10.0,
        )

        classified = monitor.classify(health)

        self.assertEqual(classified.link_state, PlatformLinkState.LOST_LINK.value)
        self.assertEqual(classified.active_lost_link_action, "climb_for_comms")

    def test_silence_triggers_return_home_policy(self) -> None:
        monitor = PlatformHealthMonitor(lost_link_timeout_s=2.0)
        health = PlatformHealthState(
            platform_id="drone-1",
            timestamp_s=10.0,
            last_seen_s=7.0,
        )

        classified = monitor.classify(health)

        self.assertEqual(classified.link_state, PlatformLinkState.LOST_LINK.value)
        self.assertEqual(classified.active_lost_link_action, "return_home")


class MissionExecutorHealthGateTest(unittest.TestCase):
    def _ctx(self, *, health: PlatformHealthState | None = None) -> MissionExecutionContext:
        return MissionExecutionContext(
            get_localization_confidence=lambda: 1.0,
            get_battery_fraction=lambda: 1.0,
            get_platform_health=lambda: health,
            plan_task=lambda task: ExecutableCommand(description=task.task_type.value),
            validate_command=lambda command: SafetyDecision(status=SafetyStatus.OK),
            execute_command=lambda command: None,
        )

    def test_lost_link_inserts_hold_before_normal_task(self) -> None:
        health = PlatformHealthState(
            platform_id="drone-1",
            timestamp_s=10.0,
            rssi_dbm=-90.0,
            last_seen_s=10.0,
        )
        state = MissionState(
            mission_id="mission-1",
            tasks=[
                MissionTask(
                    task_id="map-1",
                    task_type=MissionTaskType.MAP_FRONTIER,
                    priority=1,
                )
            ],
        )
        executor = MissionExecutor(
            state,
            MissionConstraints(geofence_radius_m=100.0),
            self._ctx(health=health),
        )

        executor.step()

        self.assertEqual(state.tasks[0].task_type, MissionTaskType.HOLD)
        self.assertEqual(state.tasks[0].status, MissionTaskStatus.PENDING)
        self.assertIn("rssi", state.tasks[0].reason or "")

    def test_low_battery_forces_return_home(self) -> None:
        ctx = MissionExecutionContext(
            get_localization_confidence=lambda: 1.0,
            get_battery_fraction=lambda: 0.1,
            plan_task=lambda task: ExecutableCommand(description=task.task_type.value),
            validate_command=lambda command: SafetyDecision(status=SafetyStatus.OK),
            execute_command=lambda command: None,
        )
        state = MissionState(mission_id="mission-1")
        executor = MissionExecutor(
            state,
            MissionConstraints(geofence_radius_m=100.0, battery_reserve_fraction=0.2),
            ctx,
        )

        executor.step()

        self.assertEqual(state.tasks[0].task_type, MissionTaskType.RETURN_HOME)
        self.assertIn("battery", state.tasks[0].reason or "")

    def test_localization_gate_inserts_single_relocalize_task(self) -> None:
        ctx = MissionExecutionContext(
            get_localization_confidence=lambda: 0.1,
            get_battery_fraction=lambda: 1.0,
            plan_task=lambda task: ExecutableCommand(description=task.task_type.value),
            validate_command=lambda command: SafetyDecision(status=SafetyStatus.OK),
            execute_command=lambda command: None,
        )
        state = MissionState(mission_id="mission-1")
        executor = MissionExecutor(state, MissionConstraints(geofence_radius_m=100.0), ctx)

        executor.step()
        executor.step()

        relocalize = [task for task in state.tasks if task.task_type == MissionTaskType.RELOCALIZE]
        self.assertEqual(len(relocalize), 1)


if __name__ == "__main__":
    unittest.main()
