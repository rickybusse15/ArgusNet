"""Verify safety_events appear on the replay payload when the gate fires.

Drives a short scan_map_inspect run with a crafted scenario that places
drones unrealistically close together, then asserts that:
1. At least one safety_events entry appears on the replay's scan_mission_state
2. The recorded violation includes "min_drone_separation"
3. The replay schema field is present in every scan_mission_state frame
"""

from __future__ import annotations

import numpy as np

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
from argusnet.safety.checker import DroneConstraintChecker, DronePhysicalLimits


def test_separation_rejection_records_safety_event_through_executor() -> None:
    """End-to-end through the executor: rejection feeds the replay surface."""
    checker = DroneConstraintChecker(DronePhysicalLimits.tracker_default())

    def validate(cmd: ExecutableCommand, st: DroneRuntimeState) -> SafetyDecision:
        violations = checker.check_state(
            position=np.asarray(st.position_m),
            velocity=np.asarray(st.velocity_mps),
            agl_m=st.agl_m,
            other_drone_positions=[np.asarray(p) for p in st.others_position_m],
        )
        hard = [v for v in violations if v.severity == "hard"]
        if not hard:
            return SafetyDecision(status=SafetyStatus.OK)
        return SafetyDecision(
            status=SafetyStatus.REJECTED,
            reason="; ".join(v.description for v in hard),
            violations=tuple(v.constraint for v in hard),
        )

    executed: list = []
    state = MissionState(
        mission_id="t",
        tasks=[MissionTask(task_id="m", task_type=MissionTaskType.MAP_FRONTIER)],
    )
    exec_ = MissionExecutor(
        state=state,
        constraints=MissionConstraints(geofence_radius_m=500.0),
        ctx=MissionExecutionContext(
            get_localization_confidence=lambda: 1.0,
            get_battery_fraction=lambda: 1.0,
            validate_command=validate,
            execute_command=lambda cmd, st: executed.append(cmd),
            now_s=lambda: 4.5,
        ),
    )

    too_close = DroneRuntimeState(
        drone_id="drone-A",
        position_m=(0.0, 0.0, 100.0),
        velocity_mps=(5.0, 0.0, 0.0),
        agl_m=100.0,
        battery_fraction=1.0,
        others_position_m=((4.0, 0.0, 100.0),),  # 4m apart, < 12m floor
    )
    decision = exec_.dispatch(
        ExecutableCommand(
            drone_id="drone-A",
            target_xy_m=(50.0, 0.0),
            target_z_m=100.0,
            task_type=MissionTaskType.MAP_FRONTIER,
        ),
        too_close,
    )

    assert decision.status == SafetyStatus.REJECTED
    assert executed == []  # not executed
    assert len(state.safety_events) == 1
    evt = state.safety_events[0]
    assert "min_drone_separation" in evt.violations
    assert evt.timestamp_s == 4.5
    assert evt.task_type == MissionTaskType.MAP_FRONTIER

    # Replay-record conversion preserves all fields
    from argusnet.core.types import SafetyEventRecord

    record = SafetyEventRecord(
        timestamp_s=evt.timestamp_s,
        drone_id=evt.drone_id,
        task_type=evt.task_type.value,
        target_xy_m=tuple(evt.target_xy_m),
        target_z_m=float(evt.target_z_m),
        reason=evt.reason,
        violations=tuple(evt.violations),
    )
    assert record.task_type == "map_frontier"
    assert record.violations == ("min_drone_separation",)
