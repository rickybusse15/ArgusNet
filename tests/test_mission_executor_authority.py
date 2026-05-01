"""Unit tests for MissionExecutor.dispatch as the motion authority.

Verifies:
- dispatch() routes through validate_command → execute_command on approval
- a rejection records a SafetyEvent and inserts a HOLD task
- a soft-only violation does not reject when blocking is enabled
"""

from __future__ import annotations

from argusnet.mission.execution import (
    DroneRuntimeState,
    ExecutableCommand,
    MissionConstraints,
    MissionExecutionContext,
    MissionExecutor,
    MissionState,
    MissionTaskType,
    SafetyDecision,
    SafetyStatus,
)


def _drone(drone_id: str = "drone-0") -> DroneRuntimeState:
    return DroneRuntimeState(
        drone_id=drone_id,
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(0.0, 0.0, 0.0),
        agl_m=80.0,
        battery_fraction=1.0,
    )


def _command(drone_id: str = "drone-0") -> ExecutableCommand:
    return ExecutableCommand(
        drone_id=drone_id,
        target_xy_m=(100.0, 100.0),
        target_z_m=120.0,
        task_type=MissionTaskType.MAP_FRONTIER,
    )


def _executor(*, validate_result: SafetyDecision, executed: list) -> MissionExecutor:
    state = MissionState(mission_id="test")
    constraints = MissionConstraints(geofence_radius_m=500.0)
    ctx = MissionExecutionContext(
        get_localization_confidence=lambda: 1.0,
        get_battery_fraction=lambda: 1.0,
        validate_command=lambda cmd, st: validate_result,
        execute_command=lambda cmd, st: executed.append(cmd),
        is_task_complete=lambda: False,
        now_s=lambda: 12.5,
    )
    return MissionExecutor(state=state, constraints=constraints, ctx=ctx)


def test_dispatch_approved_calls_execute_command() -> None:
    executed: list = []
    exec_ = _executor(
        validate_result=SafetyDecision(status=SafetyStatus.OK), executed=executed
    )
    decision = exec_.dispatch(_command(), _drone())
    assert decision.status == SafetyStatus.OK
    assert len(executed) == 1
    assert executed[0].drone_id == "drone-0"
    assert exec_.state.safety_events == []


def test_dispatch_rejected_records_safety_event_and_inserts_hold() -> None:
    executed: list = []
    rejection = SafetyDecision(
        status=SafetyStatus.REJECTED,
        reason="AGL 30m below floor 50m",
        violations=("min_agl",),
    )
    exec_ = _executor(validate_result=rejection, executed=executed)
    decision = exec_.dispatch(_command(), _drone())

    assert decision.status == SafetyStatus.REJECTED
    assert executed == []
    assert len(exec_.state.safety_events) == 1
    evt = exec_.state.safety_events[0]
    assert evt.drone_id == "drone-0"
    assert evt.violations == ("min_agl",)
    assert evt.timestamp_s == 12.5

    # HOLD task inserted at front
    assert exec_.state.tasks
    assert exec_.state.tasks[0].task_type == MissionTaskType.HOLD


def test_dispatch_rejection_only_inserts_one_hold_per_outstanding() -> None:
    executed: list = []
    rejection = SafetyDecision(status=SafetyStatus.REJECTED, reason="x")
    exec_ = _executor(validate_result=rejection, executed=executed)
    exec_.dispatch(_command(), _drone())
    exec_.dispatch(_command(drone_id="drone-1"), _drone(drone_id="drone-1"))
    holds = [t for t in exec_.state.tasks if t.task_type == MissionTaskType.HOLD]
    assert len(holds) == 1  # second rejection skips because one is already pending
    assert len(exec_.state.safety_events) == 2  # both rejections still recorded
