"""Tests for the cooperative planner module.

Covers role assignment, planning objectives, deconfliction,
replanning triggers, staleness computation, and event recording.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from smart_tracker.cooperative_planner import (
    ROLE_CORRIDOR_WATCHER,
    ROLE_PRIMARY_OBSERVER,
    ROLE_PRIORITY,
    ROLE_RELAY,
    ROLE_RESERVE,
    ROLE_SECONDARY_BASELINE,
    AltitudeProfile,
    CooperativePlanner,
    PlannerEvent,
    PlannedTrajectory,
    PlanningObjectives,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_planner(cooldown: float = 5.0) -> CooperativePlanner:
    return CooperativePlanner(PlanningObjectives(), replan_cooldown_s=cooldown)


def _stub_dynamics() -> SimpleNamespace:
    """Minimal dynamics object with attributes the planner reads."""
    return SimpleNamespace(
        drone_base_agl_m=230.0,
        follow_min_agl_m=18.0,
        terrain_following_smoothing_s=1.5,
        interceptor_follow_altitude_offset_m=35.0,
        interceptor_follow_min_agl_m=150.0,
        tracker_altitude_offset_m=70.0,
        tracker_follow_min_agl_m=155.0,
        interceptor_search_min_agl_m=180.0,
        launch_operational_altitude_agl_m=500.0,
        drone_search_speed_base_mps=28.0,
        interceptor_follow_radius_m=55.0,
        tracker_standoff_radius_m=120.0,
    )


def _stub_path_planner(route_length: float = 100.0) -> SimpleNamespace:
    """Path planner whose plan_route returns a route with a given length."""
    route = SimpleNamespace(length_m=route_length)
    return SimpleNamespace(
        plan_route=lambda start, goal, clearance_m=8.0: route,
        config=SimpleNamespace(drone_clearance_m=8.0),
    )


def _stub_path_planner_fails() -> SimpleNamespace:
    """Path planner whose plan_route always returns None."""
    return SimpleNamespace(
        plan_route=lambda start, goal, clearance_m=8.0: None,
        config=SimpleNamespace(drone_clearance_m=8.0),
    )


def _stub_terrain() -> SimpleNamespace:
    return SimpleNamespace()


# ---------------------------------------------------------------------------
# Role assignment
# ---------------------------------------------------------------------------

class TestAssignRoles:
    def test_unknown_role_replaced_with_reserve(self):
        cp = _make_planner()
        roles = cp.assign_roles(
            drone_ids=["d0", "d1"],
            target_ids=["t0"],
            drone_roles={"d0": ROLE_PRIMARY_OBSERVER, "d1": "bogus_role"},
        )
        assert roles["d1"] == ROLE_RESERVE

    def test_first_drone_elevated_if_no_primary(self):
        cp = _make_planner()
        roles = cp.assign_roles(
            drone_ids=["d0", "d1"],
            target_ids=["t0"],
            drone_roles={"d0": ROLE_RELAY, "d1": ROLE_RESERVE},
        )
        assert roles["d0"] == ROLE_PRIMARY_OBSERVER
        # A role_change event should have been recorded
        events = cp.events
        assert len(events) == 1
        assert events[0].event_type == "role_change"
        assert events[0].trigger == "forced_primary_assignment"
        assert events[0].drone_id == "d0"

    def test_no_elevation_when_primary_present(self):
        cp = _make_planner()
        roles = cp.assign_roles(
            drone_ids=["d0", "d1"],
            target_ids=["t0"],
            drone_roles={"d0": ROLE_PRIMARY_OBSERVER, "d1": ROLE_RELAY},
        )
        assert roles["d0"] == ROLE_PRIMARY_OBSERVER
        assert roles["d1"] == ROLE_RELAY
        assert len(cp.events) == 0

    def test_missing_drone_defaults_to_reserve(self):
        cp = _make_planner()
        roles = cp.assign_roles(
            drone_ids=["d0", "d1"],
            target_ids=["t0"],
            drone_roles={"d0": ROLE_PRIMARY_OBSERVER},
        )
        assert roles["d1"] == ROLE_RESERVE


# ---------------------------------------------------------------------------
# PlanningObjectives
# ---------------------------------------------------------------------------

class TestPlanningObjectives:
    def test_terrain_clearance_forced_to_one(self):
        obj = PlanningObjectives(terrain_clearance=0.0)
        assert obj.terrain_clearance == 1.0

    def test_terrain_clearance_stays_one_when_set_high(self):
        obj = PlanningObjectives(terrain_clearance=99.0)
        assert obj.terrain_clearance == 1.0

    def test_default_weights(self):
        obj = PlanningObjectives()
        assert obj.track_continuity == 1.0
        assert obj.localisation_quality == 0.8
        assert obj.terrain_clearance == 1.0


# ---------------------------------------------------------------------------
# Deconfliction
# ---------------------------------------------------------------------------

class TestCheckDeconfliction:
    def test_detects_close_pair(self):
        cp = _make_planner()
        positions = {
            "d0": np.array([0.0, 0.0, 100.0]),
            "d1": np.array([10.0, 0.0, 100.0]),
        }
        roles = {"d0": ROLE_PRIMARY_OBSERVER, "d1": ROLE_SECONDARY_BASELINE}
        violations = cp.check_deconfliction(positions, roles, min_separation_m=16.0)
        assert len(violations) == 1
        _, _, dist = violations[0]
        assert dist == pytest.approx(10.0)

    def test_no_violation_when_far_apart(self):
        cp = _make_planner()
        positions = {
            "d0": np.array([0.0, 0.0, 100.0]),
            "d1": np.array([100.0, 0.0, 100.0]),
        }
        roles = {"d0": ROLE_PRIMARY_OBSERVER, "d1": ROLE_SECONDARY_BASELINE}
        violations = cp.check_deconfliction(positions, roles, min_separation_m=16.0)
        assert violations == []

    def test_higher_priority_listed_first(self):
        cp = _make_planner()
        positions = {
            "d0": np.array([0.0, 0.0, 100.0]),
            "d1": np.array([5.0, 0.0, 100.0]),
        }
        # d0 is secondary (priority 4), d1 is primary (priority 5)
        roles = {"d0": ROLE_SECONDARY_BASELINE, "d1": ROLE_PRIMARY_OBSERVER}
        violations = cp.check_deconfliction(positions, roles, min_separation_m=16.0)
        assert len(violations) == 1
        drone_a, drone_b, _ = violations[0]
        assert drone_a == "d1"  # primary_observer (priority 5) listed first
        assert drone_b == "d0"

    def test_reserve_drones_excluded(self):
        cp = _make_planner()
        positions = {
            "d0": np.array([0.0, 0.0, 100.0]),
            "d1": np.array([1.0, 0.0, 100.0]),  # very close but reserve
        }
        roles = {"d0": ROLE_PRIMARY_OBSERVER, "d1": ROLE_RESERVE}
        violations = cp.check_deconfliction(positions, roles, min_separation_m=16.0)
        assert violations == []

    def test_only_xy_distance_matters(self):
        """Z difference should not affect deconfliction."""
        cp = _make_planner()
        positions = {
            "d0": np.array([0.0, 0.0, 0.0]),
            "d1": np.array([10.0, 0.0, 9999.0]),
        }
        roles = {"d0": ROLE_PRIMARY_OBSERVER, "d1": ROLE_CORRIDOR_WATCHER}
        violations = cp.check_deconfliction(positions, roles, min_separation_m=16.0)
        assert len(violations) == 1
        _, _, dist = violations[0]
        assert dist == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# should_replan
# ---------------------------------------------------------------------------

class TestShouldReplan:
    def _seed_plan(self, cp: CooperativePlanner, drone_id: str, role: str,
                   timestamp: float, route_length: float = 100.0):
        """Issue a plan so should_replan has state to check."""
        cp.plan_trajectory(
            drone_id=drone_id,
            role=role,
            target_position=np.array([500.0, 500.0, 0.0]),
            drone_position=np.array([0.0, 0.0, 100.0]),
            planner=_stub_path_planner(route_length),
            terrain_model=_stub_terrain(),
            dynamics=_stub_dynamics(),
            timestamp_s=timestamp,
        )

    def test_obstacle_ingress_bypasses_cooldown(self):
        cp = _make_planner(cooldown=100.0)
        self._seed_plan(cp, "d0", ROLE_PRIMARY_OBSERVER, timestamp=0.0)
        replan, reason = cp.should_replan("d0", timestamp_s=0.5, obstacle_warning=True)
        assert replan is True
        assert reason == "obstacle_ingress"

    def test_role_change_bypasses_cooldown(self):
        cp = _make_planner(cooldown=100.0)
        self._seed_plan(cp, "d0", ROLE_PRIMARY_OBSERVER, timestamp=0.0)
        replan, reason = cp.should_replan("d0", timestamp_s=0.5, role_changed=True)
        assert replan is True
        assert reason == "role_change"

    def test_track_loss_triggers_replan_for_primary(self):
        cp = _make_planner(cooldown=1.0)
        self._seed_plan(cp, "d0", ROLE_PRIMARY_OBSERVER, timestamp=0.0)
        # Primary threshold is 2 stale steps
        replan, reason = cp.should_replan(
            "d0", timestamp_s=5.0, track_stale_steps=2,
        )
        assert replan is True
        assert reason == "track_loss"

    def test_track_loss_higher_threshold_for_other_roles(self):
        cp = _make_planner(cooldown=1.0)
        self._seed_plan(cp, "d0", ROLE_CORRIDOR_WATCHER, timestamp=0.0)
        # Corridor watcher threshold is 6; 3 stale steps should NOT trigger
        replan, _ = cp.should_replan("d0", timestamp_s=5.0, track_stale_steps=3)
        assert replan is False
        # At 6, it should trigger
        replan, reason = cp.should_replan("d0", timestamp_s=5.0, track_stale_steps=6)
        assert replan is True
        assert reason == "track_loss"

    def test_staleness_expiry(self):
        cp = _make_planner(cooldown=1.0)
        self._seed_plan(cp, "d0", ROLE_PRIMARY_OBSERVER, timestamp=0.0,
                        route_length=100.0)
        # compute_staleness(0, 100, 28) = 0 + max(30, 100/28*0.5) = 30.0
        # After 30s the plan is stale
        replan, reason = cp.should_replan("d0", timestamp_s=31.0)
        assert replan is True
        assert reason == "staleness_expiry"

    def test_cooldown_prevents_replan(self):
        cp = _make_planner(cooldown=10.0)
        self._seed_plan(cp, "d0", ROLE_PRIMARY_OBSERVER, timestamp=0.0)
        # Within cooldown, no non-bypass trigger fires
        replan, reason = cp.should_replan("d0", timestamp_s=3.0)
        assert replan is False
        assert reason == ""

    def test_no_plan_returns_false(self):
        """A drone with no existing plan and no triggers returns False."""
        cp = _make_planner()
        replan, reason = cp.should_replan("unknown_drone", timestamp_s=100.0)
        assert replan is False


# ---------------------------------------------------------------------------
# compute_staleness
# ---------------------------------------------------------------------------

class TestComputeStaleness:
    def test_formula_minimum_30s(self):
        cp = _make_planner()
        # route_length/speed*0.5 = 50/10*0.5 = 2.5 < 30, so window = 30
        result = cp.compute_staleness(planned_at_s=10.0, route_length_m=50.0,
                                      speed_mps=10.0)
        assert result == pytest.approx(40.0)

    def test_formula_long_route(self):
        cp = _make_planner()
        # route_length/speed*0.5 = 2000/10*0.5 = 100 > 30, so window = 100
        result = cp.compute_staleness(planned_at_s=5.0, route_length_m=2000.0,
                                      speed_mps=10.0)
        assert result == pytest.approx(105.0)

    def test_zero_speed_does_not_crash(self):
        cp = _make_planner()
        # speed clamped to 1e-6 internally to avoid division by zero
        result = cp.compute_staleness(planned_at_s=0.0, route_length_m=100.0,
                                      speed_mps=0.0)
        assert np.isfinite(result)
        assert result > 0.0


# ---------------------------------------------------------------------------
# plan_trajectory
# ---------------------------------------------------------------------------

class TestPlanTrajectory:
    def test_successful_plan_records_event(self):
        cp = _make_planner()
        traj = cp.plan_trajectory(
            drone_id="d0",
            role=ROLE_PRIMARY_OBSERVER,
            target_position=np.array([500.0, 500.0, 0.0]),
            drone_position=np.array([0.0, 0.0, 100.0]),
            planner=_stub_path_planner(200.0),
            terrain_model=_stub_terrain(),
            dynamics=_stub_dynamics(),
            timestamp_s=10.0,
        )
        assert isinstance(traj, PlannedTrajectory)
        assert traj.drone_id == "d0"
        assert traj.role == ROLE_PRIMARY_OBSERVER
        assert traj.route is not None
        assert traj.generation == 1
        events = cp.events
        assert any(e.event_type == "plan_issued" for e in events)

    def test_failed_plan_records_rejection(self):
        cp = _make_planner()
        traj = cp.plan_trajectory(
            drone_id="d0",
            role=ROLE_PRIMARY_OBSERVER,
            target_position=np.array([500.0, 500.0, 0.0]),
            drone_position=np.array([0.0, 0.0, 100.0]),
            planner=_stub_path_planner_fails(),
            terrain_model=_stub_terrain(),
            dynamics=_stub_dynamics(),
            timestamp_s=10.0,
        )
        assert traj.route is None
        assert traj.override_reason == "path_planner_returned_none"
        events = cp.events
        assert any(e.event_type == "plan_rejected" for e in events)

    def test_reserve_drone_gets_no_route(self):
        cp = _make_planner()
        traj = cp.plan_trajectory(
            drone_id="d0",
            role=ROLE_RESERVE,
            target_position=np.array([500.0, 500.0, 0.0]),
            drone_position=np.array([0.0, 0.0, 100.0]),
            planner=_stub_path_planner(100.0),
            terrain_model=_stub_terrain(),
            dynamics=_stub_dynamics(),
            timestamp_s=0.0,
        )
        assert traj.route is None

    def test_generation_increments(self):
        cp = _make_planner()
        t1 = cp.plan_trajectory(
            "d0", ROLE_PRIMARY_OBSERVER, np.array([100, 100, 0]),
            np.array([0, 0, 100]), _stub_path_planner(), _stub_terrain(),
            _stub_dynamics(), 0.0,
        )
        t2 = cp.plan_trajectory(
            "d0", ROLE_PRIMARY_OBSERVER, np.array([100, 100, 0]),
            np.array([0, 0, 100]), _stub_path_planner(), _stub_terrain(),
            _stub_dynamics(), 5.0,
        )
        assert t2.generation == t1.generation + 1


# ---------------------------------------------------------------------------
# Event recording
# ---------------------------------------------------------------------------

class TestEventRecording:
    def test_replan_trigger_events_recorded(self):
        cp = _make_planner(cooldown=1.0)
        # Seed a plan
        cp.plan_trajectory(
            "d0", ROLE_PRIMARY_OBSERVER, np.array([500, 500, 0]),
            np.array([0, 0, 100]), _stub_path_planner(), _stub_terrain(),
            _stub_dynamics(), 0.0,
        )
        initial_count = len(cp.events)
        cp.should_replan("d0", timestamp_s=5.0, obstacle_warning=True)
        assert len(cp.events) == initial_count + 1
        last = cp.events[-1]
        assert last.event_type == "replan_trigger"
        assert last.trigger == "obstacle_ingress"

    def test_role_change_event_on_assign(self):
        cp = _make_planner()
        cp.assign_roles(["d0"], ["t0"], {"d0": ROLE_RELAY})
        events = [e for e in cp.events if e.event_type == "role_change"]
        assert len(events) == 1
        assert events[0].trigger == "forced_primary_assignment"
