"""Tests for the curated in-range ``target_tracking`` demo scenario.

The demo (``argusnet sim --demo tracking`` / ``tracking_demo_options()``) exists
so a first-time run produces confirmed fused tracks instead of the expected-but-
confusing zero-track result on the default large map. The acceptance criterion is
that the pipeline confirms at least one fused track.
"""

from __future__ import annotations

import argparse
import unittest

from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    TRACKING_DEMO_DEFAULTS,
    add_cli_arguments,
    build_default_scenario,
    run_simulation,
    tracking_demo_options,
)


def _max_confirmed_tracks(options: ScenarioOptions, *, steps: int = 60) -> int:
    scenario = build_default_scenario(options=options, seed=7)
    result = run_simulation(scenario, SimulationConfig(steps=steps, dt_s=0.25, seed=7))
    return max((f.metrics.active_track_count for f in result.frames), default=0)


class TrackingDemoOptionsTest(unittest.TestCase):
    def test_factory_bundle_is_target_tracking_and_in_range(self) -> None:
        options = tracking_demo_options()
        self.assertEqual(options.mission_mode, "target_tracking")
        # A small footprint keeps targets inside drone sensor range.
        self.assertEqual(options.map_preset, "small")
        self.assertGreaterEqual(options.target_count, 1)

    def test_factory_overrides_apply(self) -> None:
        options = tracking_demo_options(target_count=1, map_preset="medium")
        self.assertEqual(options.target_count, 1)
        self.assertEqual(options.map_preset, "medium")
        # Non-overridden curated fields are preserved.
        self.assertEqual(options.mission_mode, "target_tracking")

    def test_demo_scenario_confirms_fused_tracks(self) -> None:
        """Acceptance: the curated demo produces at least one confirmed track."""
        confirmed = _max_confirmed_tracks(tracking_demo_options())
        self.assertGreaterEqual(
            confirmed,
            1,
            "curated tracking demo should confirm at least one fused track",
        )


class TrackingDemoCliWiringTest(unittest.TestCase):
    @staticmethod
    def _parse(argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        add_cli_arguments(parser)
        return parser.parse_args(argv)

    def _apply_demo(self, args: argparse.Namespace) -> argparse.Namespace:
        # Mirror the demo-application block in run_from_args without running a sim.
        from argusnet.simulation.sim import _DEMO_FIELD_TO_ARG_DEST, _arg_was_provided

        if getattr(args, "demo", None) == "tracking":
            for field_name, value in TRACKING_DEMO_DEFAULTS.items():
                dest = _DEMO_FIELD_TO_ARG_DEST[field_name]
                if not _arg_was_provided(args, dest):
                    setattr(args, dest, value)
        return args

    def test_demo_flag_seeds_curated_defaults(self) -> None:
        args = self._apply_demo(self._parse(["--demo", "tracking"]))
        self.assertEqual(args.mission_mode, "target_tracking")
        self.assertEqual(args.map_preset, "small")
        self.assertEqual(args.target_count, TRACKING_DEMO_DEFAULTS["target_count"])

    def test_explicit_flag_overrides_demo(self) -> None:
        args = self._apply_demo(
            self._parse(["--demo", "tracking", "--drone-count", "6", "--map-preset", "medium"])
        )
        # Explicitly-provided flags win over the curated bundle.
        self.assertEqual(args.drone_count, 6)
        self.assertEqual(args.map_preset, "medium")
        # Unspecified curated fields still applied.
        self.assertEqual(args.mission_mode, "target_tracking")

    def test_no_demo_leaves_defaults(self) -> None:
        args = self._apply_demo(self._parse([]))
        self.assertIsNone(args.demo)
        self.assertEqual(args.mission_mode, "scan_map_inspect")
        self.assertEqual(args.map_preset, "regional")


if __name__ == "__main__":
    unittest.main()
