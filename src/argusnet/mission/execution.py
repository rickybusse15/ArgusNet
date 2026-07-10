from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from argusnet.core.types import PlatformHealthState, PlatformLinkState
from argusnet.mission.health import PlatformHealthMonitor

# -----------------------------
# Core enums and data classes
# -----------------------------


class MissionStatus(str, Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    RETURNING_HOME = "returning_home"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class MissionTaskType(str, Enum):
    MAP_FRONTIER = "map_frontier"
    INSPECT_TARGET = "inspect_target"
    INSPECT_POI = "inspect_poi"
    RELOCALIZE = "relocalize"
    REVISIT = "revisit"
    RETURN_HOME = "return_home"
    HOLD = "hold"
    OPERATOR_REVIEW = "operator_review"


class MissionTaskStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    BLOCKED = "blocked"
    FAILED = "failed"


class SafetyStatus(str, Enum):
    OK = "ok"
    REJECTED = "rejected"


@dataclass
class MissionConstraints:
    geofence_radius_m: float
    battery_reserve_fraction: float = 0.2
    localization_confidence_min: float = 0.5
    comms_required: bool = True


@dataclass
class MissionTask:
    task_id: str
    task_type: MissionTaskType
    priority: int = 1
    status: MissionTaskStatus = MissionTaskStatus.PENDING
    reason: str | None = None


@dataclass
class ExecutableCommand:
    description: str


@dataclass
class SafetyDecision:
    status: SafetyStatus
    reason: str | None = None


@dataclass
class MissionState:
    mission_id: str
    status: MissionStatus = MissionStatus.CREATED
    tasks: list[MissionTask] = field(default_factory=list)
    active_task: MissionTask | None = None


# -----------------------------
# Execution context (dependency injection)
# -----------------------------


@dataclass
class MissionExecutionContext:
    get_localization_confidence: Callable[[], float]
    get_battery_fraction: Callable[[], float]
    plan_task: Callable[[MissionTask], ExecutableCommand]
    validate_command: Callable[[ExecutableCommand], SafetyDecision]
    execute_command: Callable[[ExecutableCommand], None]
    is_task_complete: Callable[[], bool] = field(default_factory=lambda: lambda: True)
    get_platform_health: Callable[[], PlatformHealthState | None] | None = None
    on_health_update: Callable[[PlatformHealthState], None] | None = None


# -----------------------------
# Mission executor
# -----------------------------


class MissionExecutor:
    """
    Lightweight closed-loop mission execution skeleton.

    This class does not implement mapping, localization, or planning logic.
    It orchestrates them via injected callables.
    """

    def __init__(
        self,
        state: MissionState,
        constraints: MissionConstraints,
        ctx: MissionExecutionContext,
    ):
        self.state = state
        self.constraints = constraints
        self.ctx = ctx
        self.health_monitor = PlatformHealthMonitor(
            battery_return_fraction=constraints.battery_reserve_fraction
        )

    # -------------------------
    # Core loop step
    # -------------------------

    def step(self) -> None:
        """Execute one iteration of the mission loop."""

        terminal_states = {MissionStatus.COMPLETED, MissionStatus.ABORTED, MissionStatus.FAILED}
        if self.state.status in terminal_states:
            return

        # 1. Ensure active state
        if self.state.status == MissionStatus.CREATED:
            self.state.status = MissionStatus.INITIALIZING

        if self.state.status == MissionStatus.INITIALIZING:
            self.state.status = MissionStatus.ACTIVE

        # 2. Safety: battery check
        battery = self.ctx.get_battery_fraction()
        if battery < self.constraints.battery_reserve_fraction:
            self._force_return_home("battery low")
            return

        # 3. Data-link / platform health gate
        if self.constraints.comms_required:
            health = self._current_health()
            if health is not None:
                if health.link_state == PlatformLinkState.RETURNING_HOME.value:
                    self._force_return_home(health.reason or "platform health requires return-home")
                    return
                if self.health_monitor.should_hold_or_return(health):
                    self._activate_hold(health.reason or "platform health degraded")
                    return

        # 4. Localization gate
        localization_conf = self.ctx.get_localization_confidence()
        if localization_conf < self.constraints.localization_confidence_min:
            self._activate_relocalization()
            return

        # 5. Select task
        task = self._select_task()
        if task is None:
            self.state.status = MissionStatus.COMPLETED
            return

        self.state.active_task = task
        task.status = MissionTaskStatus.ACTIVE

        # 6. Plan
        command = self.ctx.plan_task(task)

        # 7. Safety validation
        decision = self.ctx.validate_command(command)
        if decision.status == SafetyStatus.REJECTED:
            task.status = MissionTaskStatus.BLOCKED
            task.reason = decision.reason
            return

        # 8. Execute
        self.ctx.execute_command(command)

        # 9. Mark complete when the injected completion check confirms it
        if self.ctx.is_task_complete():
            task.status = MissionTaskStatus.COMPLETE

    # -------------------------
    # Helpers
    # -------------------------

    def _select_task(self) -> MissionTask | None:
        pending = [t for t in self.state.tasks if t.status == MissionTaskStatus.PENDING]
        if not pending:
            return None
        return sorted(pending, key=lambda t: -t.priority)[0]

    def _force_return_home(self, reason: str) -> None:
        self.state.status = MissionStatus.RETURNING_HOME
        self.state.tasks = [
            MissionTask(
                task_id="return_home",
                task_type=MissionTaskType.RETURN_HOME,
                priority=100,
                reason=reason,
            )
        ]

    def _activate_relocalization(self) -> None:
        if any(
            t.task_type == MissionTaskType.RELOCALIZE
            and t.status in {MissionTaskStatus.PENDING, MissionTaskStatus.ACTIVE}
            for t in self.state.tasks
        ):
            return
        self.state.tasks.insert(
            0,
            MissionTask(task_id="relocalize", task_type=MissionTaskType.RELOCALIZE, priority=100),
        )

    def _activate_hold(self, reason: str) -> None:
        if any(
            t.task_type == MissionTaskType.HOLD
            and t.status in {MissionTaskStatus.PENDING, MissionTaskStatus.ACTIVE}
            for t in self.state.tasks
        ):
            return
        self.state.tasks.insert(
            0,
            MissionTask(
                task_id="hold_for_health",
                task_type=MissionTaskType.HOLD,
                priority=95,
                reason=reason,
            ),
        )

    def _current_health(self) -> PlatformHealthState | None:
        if self.ctx.get_platform_health is None:
            return None
        raw = self.ctx.get_platform_health()
        if raw is None:
            return None
        classified = self.health_monitor.classify(raw)
        if self.ctx.on_health_update is not None:
            self.ctx.on_health_update(classified)
        return classified
