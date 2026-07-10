"""Platform health and lost-link policy helpers."""

from __future__ import annotations

from dataclasses import replace

from argusnet.core.types import LostLinkAction, PlatformHealthState, PlatformLinkState


class PlatformHealthMonitor:
    """Classify data-link health and choose conservative lost-link actions."""

    def __init__(
        self,
        *,
        min_rssi_dbm: float = -85.0,
        degraded_margin_db: float = 6.0,
        lost_link_timeout_s: float = 2.0,
        stale_frame_limit: int = 3,
        battery_return_fraction: float = 0.2,
    ) -> None:
        self.min_rssi_dbm = min_rssi_dbm
        self.degraded_margin_db = degraded_margin_db
        self.lost_link_timeout_s = lost_link_timeout_s
        self.stale_frame_limit = stale_frame_limit
        self.battery_return_fraction = battery_return_fraction

    def classify(
        self,
        health: PlatformHealthState,
        *,
        now_s: float | None = None,
    ) -> PlatformHealthState:
        timestamp_s = health.timestamp_s if now_s is None else now_s
        last_seen_s = health.last_seen_s if health.last_seen_s is not None else health.timestamp_s
        silence_s = max(0.0, timestamp_s - last_seen_s)

        state = PlatformLinkState.NOMINAL.value
        reason = None
        action = None

        if health.battery_fraction <= self.battery_return_fraction:
            state = PlatformLinkState.RETURNING_HOME.value
            action = LostLinkAction.RETURN_HOME.value
            reason = "battery reserve reached"
        elif (
            silence_s >= self.lost_link_timeout_s
            or health.stale_frame_count > self.stale_frame_limit
        ):
            state = PlatformLinkState.LOST_LINK.value
            action = LostLinkAction.RETURN_HOME.value
            reason = "data link timed out"
        elif health.rssi_dbm is not None and health.rssi_dbm < self.min_rssi_dbm:
            state = PlatformLinkState.LOST_LINK.value
            action = LostLinkAction.CLIMB_FOR_COMMS.value
            reason = "rssi below lost-link threshold"
        elif (
            health.rssi_dbm is not None
            and health.rssi_dbm < self.min_rssi_dbm + self.degraded_margin_db
        ):
            state = PlatformLinkState.DEGRADED.value
            action = LostLinkAction.HOLD.value
            reason = "rssi degraded"
        elif health.link_state == PlatformLinkState.LOST_LINK.value:
            state = PlatformLinkState.REACQUIRING.value
            action = LostLinkAction.HOLD.value
            reason = "link restored, reacquiring"

        return replace(
            health,
            timestamp_s=timestamp_s,
            link_state=state,
            active_lost_link_action=action,
            reason=reason,
        )

    def should_hold_or_return(self, health: PlatformHealthState) -> bool:
        return health.link_state in {
            PlatformLinkState.DEGRADED.value,
            PlatformLinkState.LOST_LINK.value,
            PlatformLinkState.REACQUIRING.value,
            PlatformLinkState.RETURNING_HOME.value,
        }
