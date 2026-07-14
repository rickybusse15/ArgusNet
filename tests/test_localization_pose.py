"""Runtime localization pose/covariance/status contract (LOCALIZATION.md §7/§8/§13).

Milestone: expose a richer pose/covariance/status interface (LocalizationQuery)
before precision inspection routing and safety depend on it.
"""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.types import LocalizationStatus, PoseEstimate
from argusnet.localization import (
    LOCALIZATION_QUERY_CONTRACT_VERSION,
    GridLocalizer,
    LocalizationConfig,
    LocalizationQuery,
)
from argusnet.localization.engine import Z_HOLD_STD_M
from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


def _fully_covered(bounds: GridBounds) -> CoverageMap:
    cov = CoverageMap(bounds)
    cov.mark_circular((0.0, 0.0), radius_m=10_000.0)  # covers the whole grid
    return cov


def _drive(localizer: GridLocalizer, cov: CoverageMap, steps: int, drone_id: str = "d0"):
    est = None
    for k in range(steps):
        est = localizer.update(
            drone_id=drone_id,
            nominal_position=np.array([0.0, 0.0, 80.0]),
            heading_rad=0.3,
            coverage_map=cov,
            footprint_radius_m=60.0,
            timestamp_s=float(k),
        )
    return est


class ContractTests(unittest.TestCase):
    def test_gridlocalizer_satisfies_protocol_and_provenance(self) -> None:
        loc = GridLocalizer()
        self.assertIsInstance(loc, LocalizationQuery)
        self.assertEqual(loc.source_id, "grid")
        self.assertEqual(loc.version, "1.0")
        self.assertEqual(LOCALIZATION_QUERY_CONTRACT_VERSION, "1.0")

    def test_unknown_platform_is_unlocalized_with_no_pose(self) -> None:
        loc = GridLocalizer()
        self.assertIsNone(loc.current_pose("ghost"))
        self.assertEqual(loc.localization_status("ghost"), LocalizationStatus.UNLOCALIZED.value)
        self.assertEqual(loc.confidence("ghost"), 0.0)
        self.assertFalse(loc.is_localized("ghost"))
        self.assertEqual(loc.current_covariance("ghost"), ())


class CovarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = GridBounds(-300.0, 300.0, -300.0, 300.0, 30.0)

    def test_covariance_symmetric_psd_with_altitude_hold_term(self) -> None:
        loc = GridLocalizer()
        _drive(loc, _fully_covered(self.bounds), steps=8)
        pose = loc.current_pose("d0")
        self.assertIsNotNone(pose)
        cov = np.array(pose.covariance, dtype=float)
        self.assertEqual(cov.size, 9)
        m = cov.reshape(3, 3)
        np.testing.assert_allclose(m, m.T, atol=1e-9)  # symmetric
        self.assertGreaterEqual(float(np.linalg.eigvalsh(m).min()), -1e-9)  # PSD
        self.assertAlmostEqual(m[2, 2], Z_HOLD_STD_M**2)  # altitude-hold term
        self.assertGreater(m[0, 0] + m[1, 1], 0.0)  # real xy spread

    def test_covariance_shrinks_as_localization_converges(self) -> None:
        loc = GridLocalizer()
        cov = _fully_covered(self.bounds)
        _drive(loc, cov, steps=2)
        early = np.array(loc.current_covariance("d0")).reshape(3, 3)[:2, :2]
        _drive(loc, cov, steps=25)
        late = np.array(loc.current_covariance("d0")).reshape(3, 3)[:2, :2]
        self.assertLess(np.trace(late), np.trace(early))  # annealing tightens the cloud


class StatusModelTests(unittest.TestCase):
    def test_low_coverage_is_unlocalized(self) -> None:
        loc = GridLocalizer()
        bounds = GridBounds(-300.0, 300.0, -300.0, 300.0, 30.0)
        cov = CoverageMap(bounds)  # empty -> coverage below min_coverage_to_localize
        _drive(loc, cov, steps=5)
        self.assertEqual(loc.localization_status("d0"), LocalizationStatus.UNLOCALIZED.value)
        self.assertFalse(loc.is_localized("d0"))

    def test_high_coverage_converges_to_localized(self) -> None:
        loc = GridLocalizer()
        bounds = GridBounds(-300.0, 300.0, -300.0, 300.0, 30.0)
        _drive(loc, _fully_covered(bounds), steps=30)
        self.assertEqual(loc.localization_status("d0"), LocalizationStatus.LOCALIZED.value)
        self.assertTrue(loc.is_localized("d0"))
        self.assertTrue(loc.is_localized("d0", threshold=0.5))

    def test_classifier_degraded_then_lost(self) -> None:
        loc = GridLocalizer(LocalizationConfig())
        common = dict(coverage_fraction=0.9, timed_out=False)
        # A previously localized fix whose confidence dips -> degraded.
        degraded = loc._classify_status(
            LocalizationStatus.LOCALIZED.value, confidence=0.3, position_std=40.0, **common
        )
        self.assertEqual(degraded, LocalizationStatus.DEGRADED.value)
        # Then collapses entirely -> lost.
        lost = loc._classify_status(
            LocalizationStatus.DEGRADED.value, confidence=0.02, position_std=200.0, **common
        )
        self.assertEqual(lost, LocalizationStatus.LOST.value)

    def test_classifier_initializing_vs_unlocalized_and_timeout(self) -> None:
        loc = GridLocalizer(LocalizationConfig())
        common = dict(coverage_fraction=0.5, timed_out=False)
        init = loc._classify_status(
            LocalizationStatus.UNLOCALIZED.value, confidence=0.4, position_std=40.0, **common
        )
        self.assertEqual(init, LocalizationStatus.INITIALIZING.value)
        still = loc._classify_status(
            LocalizationStatus.UNLOCALIZED.value, confidence=0.01, position_std=70.0, **common
        )
        self.assertEqual(still, LocalizationStatus.UNLOCALIZED.value)
        forced = loc._classify_status(
            LocalizationStatus.UNLOCALIZED.value,
            confidence=0.0,
            position_std=70.0,
            coverage_fraction=0.02,
            timed_out=True,
        )
        self.assertEqual(forced, LocalizationStatus.LOCALIZED.value)

    def test_failure_reason_set_for_degraded(self) -> None:
        loc = GridLocalizer()
        # Force a degraded status directly, then build a pose from stored state.
        _drive(loc, _fully_covered(GridBounds(-300, 300, -300, 300, 30.0)), steps=6)
        loc._statuses["d0"] = LocalizationStatus.DEGRADED.value
        pose = loc.current_pose("d0")
        self.assertEqual(pose.status, LocalizationStatus.DEGRADED.value)
        self.assertIsNotNone(pose.failure_reason)


class RuntimeSurfacingTests(unittest.TestCase):
    def _run(self, duration_s: float = 30.0):
        scenario = build_default_scenario(
            options=ScenarioOptions(
                map_preset="small",
                terrain_preset="alpine",
                drone_count=3,
                mission_mode="scan_map_inspect",
                coverage_resolution_m=30.0,
            ),
            seed=7,
        )
        return run_simulation(
            scenario=scenario,
            simulation_config=SimulationConfig.from_duration(duration_s, dt_s=0.5, seed=7),
            tracker_config=None,
        )

    def test_pose_estimates_surface_alongside_localization_estimates(self) -> None:
        result = self._run()
        checked = 0
        valid = {s.value for s in LocalizationStatus}
        for frame in result.frames:
            sm = frame.scan_mission_state
            if sm is None or not sm.pose_estimates:
                continue
            # Same platforms, same order as the scalar estimates.
            self.assertEqual(
                [p.platform_id for p in sm.pose_estimates],
                [e.drone_id for e in sm.localization_estimates],
            )
            for p in sm.pose_estimates:
                self.assertIsInstance(p, PoseEstimate)
                self.assertEqual(len(p.covariance), 9)
                self.assertIn(p.status, valid)
                checked += 1
        self.assertGreater(checked, 0)

    def test_localized_status_reached_in_longer_run(self) -> None:
        result = self._run(duration_s=120.0)
        seen = {
            p.status
            for frame in result.frames
            if frame.scan_mission_state is not None
            for p in frame.scan_mission_state.pose_estimates
        }
        self.assertIn(LocalizationStatus.LOCALIZED.value, seen)

    def test_pose_estimates_deterministic(self) -> None:
        def poses(res):
            return [
                tuple(
                    (p.platform_id, p.status, tuple(round(c, 4) for c in p.covariance))
                    for p in f.scan_mission_state.pose_estimates
                )
                for f in res.frames
                if f.scan_mission_state is not None
            ]

        self.assertEqual(poses(self._run()), poses(self._run()))

    def test_poses_not_surfaced_after_localization_stops(self) -> None:
        # Once the mission leaves scanning/localizing the localizer stops
        # refreshing; stale poses must not be replayed as "current".
        result = self._run(duration_s=120.0)
        frames = [f.scan_mission_state for f in result.frames if f.scan_mission_state is not None]
        self.assertTrue(any(sm.pose_estimates for sm in frames), "some frames should surface poses")
        post_localization_empty = [
            sm
            for sm in frames
            if sm.phase in ("inspecting", "egress", "complete") and not sm.pose_estimates
        ]
        self.assertTrue(post_localization_empty, "post-localization frames must not replay poses")
        self.assertEqual(frames[-1].pose_estimates, ())  # no frozen pose lingers to the end


if __name__ == "__main__":
    unittest.main()
