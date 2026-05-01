"""Verify the dormant coordination primitives are now driven from sim.py.

Phase C3 of the post-Phase-B follow-up plan: ``CoordinationManager``'s
``update_claimed`` / ``flush_messages`` / ``formation_offsets`` and
``FrontierPlanner.select_frontier_cell`` were all defined and tested in
isolation but not invoked by the runtime. This test asserts they are now
called during a multi-drone scan_map_inspect run, and that
``select_frontier_cell``'s exclusion-radius contract still holds.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds
from argusnet.planning.coordination import CoordinationManager
from argusnet.planning.frontier import ClaimedCells, FrontierConfig, FrontierPlanner
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    run_simulation,
)


class CoordinationCallsObservedTest(unittest.TestCase):
    """Patch the coordination methods and confirm sim.py invokes them."""

    def _short_scan_run(self, *, mock_target: str):
        opts = ScenarioOptions(
            mission_mode="scan_map_inspect",
            drone_count=2,
            scan_coverage_threshold=0.95,  # ensures we stay in scanning
            poi_count=2,
            target_count=0,
        )
        scenario = build_default_scenario(options=opts, seed=13)
        with patch.object(
            CoordinationManager, mock_target, autospec=True
        ) as mocked:
            mocked.return_value = None
            run_simulation(scenario, SimulationConfig(steps=4, dt_s=0.5, seed=13))
        return mocked

    def test_update_claimed_is_invoked(self) -> None:
        mocked = self._short_scan_run(mock_target="update_claimed")
        self.assertGreater(
            mocked.call_count,
            0,
            "CoordinationManager.update_claimed must be called from sim.py",
        )

    def test_flush_messages_is_invoked(self) -> None:
        mocked = self._short_scan_run(mock_target="flush_messages")
        self.assertGreater(
            mocked.call_count,
            0,
            "CoordinationManager.flush_messages must be called from sim.py",
        )

    def test_formation_offsets_is_invoked(self) -> None:
        # Configure the mock to return the empty dict so default-config
        # behaviour is preserved.
        mocked = self._short_scan_run(mock_target="formation_offsets")
        self.assertGreater(
            mocked.call_count,
            0,
            "CoordinationManager.formation_offsets must be called from sim.py",
        )


class FrontierExclusionContractTest(unittest.TestCase):
    """The select_frontier_cell exclusion radius is what keeps drones apart."""

    def _empty_coverage(self, span_m: float = 300.0, resolution_m: float = 10.0) -> CoverageMap:
        bounds = GridBounds(
            x_min_m=0.0,
            x_max_m=span_m,
            y_min_m=0.0,
            y_max_m=span_m,
            resolution_m=resolution_m,
        )
        return CoverageMap(bounds=bounds)

    def test_second_drone_pick_avoids_first_drones_claim(self) -> None:
        cmap = self._empty_coverage()
        claimed = ClaimedCells()
        planner = FrontierPlanner(
            FrontierConfig(exclusion_radius_cells=4)
        )

        # Drone-A picks first, with no claims. Then claim its choice.
        pick_a = planner.select_frontier_cell(
            cmap, np.array([50.0, 50.0]), claimed, "drone-a"
        )
        self.assertIsNotNone(pick_a)
        claimed.claim("drone-a", pick_a)

        # Drone-B picks second from a similar starting position. Its choice
        # must be outside the exclusion radius around drone-A's claim.
        pick_b = planner.select_frontier_cell(
            cmap, np.array([60.0, 50.0]), claimed, "drone-b"
        )
        self.assertIsNotNone(pick_b)
        di = abs(pick_a[0] - pick_b[0])
        dj = abs(pick_a[1] - pick_b[1])
        self.assertFalse(
            di <= 4 and dj <= 4,
            f"drone-b cell {pick_b} is within exclusion radius of drone-a {pick_a}",
        )


class CoordinationManagerLatencyTest(unittest.TestCase):
    """Default config (latency=0) applies claims immediately; sim relies on this."""

    def test_zero_latency_applies_claim_immediately(self) -> None:
        from argusnet.planning.coordination import (
            CoordinationPolicy,
            SharedMissionState,
        )

        manager = CoordinationManager(CoordinationPolicy())
        shared = SharedMissionState()
        manager.update_claimed("drone-1", (3, 7), shared, current_step=0)
        self.assertEqual(shared.claimed_cells.get("drone-1"), (3, 7))

    def test_flush_messages_is_a_noop_with_no_pending(self) -> None:
        from argusnet.planning.coordination import (
            CoordinationPolicy,
            SharedMissionState,
        )

        manager = CoordinationManager(CoordinationPolicy())
        shared = SharedMissionState()
        manager.flush_messages(shared, current_step=0)  # must not raise
        self.assertEqual(dict(shared.pending_messages), {})


if __name__ == "__main__":
    unittest.main()
