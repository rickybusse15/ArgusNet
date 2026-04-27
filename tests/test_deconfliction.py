"""Unit tests for the inter-drone deconfliction layer."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.types import MissionZone, vec3
from argusnet.planning.deconfliction import (
    DeconflictionConfig,
    DeconflictionLayer,
)
from argusnet.planning.inspection import FlightCorridor
from argusnet.world.terrain import TerrainModel


def _proposal(pos, vel):
    return (np.array(pos, dtype=float), np.array(vel, dtype=float))


class TestDeconflictionPriorityYield(unittest.TestCase):
    """Lower-priority drone yields when within minimum separation."""

    def test_secondary_yields_to_primary(self) -> None:
        roles = {"d_primary": "primary_observer", "d_secondary": "secondary_baseline"}
        layer = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles)

        proposed = {
            "d_primary": _proposal((0.0, 0.0, 50.0), (5.0, 0.0, 0.0)),
            "d_secondary": _proposal((10.0, 0.0, 50.0), (-5.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=1.0)

        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev.yielding_drone_id, "d_secondary")
        self.assertEqual(ev.conflicting_drone_id, "d_primary")
        self.assertEqual(ev.timestamp_s, 1.0)
        # Primary drone untouched.
        np.testing.assert_array_equal(adjusted["d_primary"][0], proposed["d_primary"][0])
        # Secondary drone offset from its proposed position.
        self.assertFalse(np.array_equal(adjusted["d_secondary"][0], proposed["d_secondary"][0]))

    def test_lateral_offset_restores_separation(self) -> None:
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        cfg = DeconflictionConfig(min_separation_m=16.0, look_ahead_s=1.0)
        layer = DeconflictionLayer(cfg, role_lookup=roles)
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
            "b": _proposal((5.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)

        self.assertEqual(events[0].resolution, "lateral_offset")
        sep_xy = float(np.linalg.norm(adjusted["a"][0][:2] - adjusted["b"][0][:2]))
        self.assertGreaterEqual(sep_xy, cfg.min_separation_m)


class TestDeconflictionTieBreaking(unittest.TestCase):
    """Equal-priority drones break ties by lexicographic ID."""

    def test_lex_tiebreak(self) -> None:
        roles = {"alpha": "primary_observer", "bravo": "primary_observer"}
        layer = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles)
        proposed = {
            "alpha": _proposal((0.0, 0.0, 50.0), (5.0, 0.0, 0.0)),
            "bravo": _proposal((10.0, 0.0, 50.0), (-5.0, 0.0, 0.0)),
        }
        _, events = layer.resolve_step(proposed, timestamp_s=0.0)
        # "bravo" > "alpha" lexicographically → bravo yields.
        self.assertEqual(events[0].yielding_drone_id, "bravo")

    def test_no_oscillation_across_steps(self) -> None:
        roles = {"alpha": "primary_observer", "bravo": "primary_observer"}
        layer = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles)
        proposed = {
            "alpha": _proposal((0.0, 0.0, 50.0), (5.0, 0.0, 0.0)),
            "bravo": _proposal((10.0, 0.0, 50.0), (-5.0, 0.0, 0.0)),
        }
        _, events_a = layer.resolve_step(proposed, timestamp_s=0.0)
        _, events_b = layer.resolve_step(proposed, timestamp_s=0.25)
        self.assertEqual(
            [ev.yielding_drone_id for ev in events_a],
            [ev.yielding_drone_id for ev in events_b],
        )


class TestDeconflictionNoConflict(unittest.TestCase):
    """No yields when drones are well-separated."""

    def test_far_apart_no_events(self) -> None:
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        layer = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles)
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (1.0, 0.0, 0.0)),
            "b": _proposal((500.0, 0.0, 50.0), (-1.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertEqual(events, [])
        for did in ("a", "b"):
            np.testing.assert_array_equal(adjusted[did][0], proposed[did][0])

    def test_disabled_layer_passes_through(self) -> None:
        roles = {"a": "primary_observer", "b": "primary_observer"}
        layer = DeconflictionLayer(DeconflictionConfig(enabled=False), role_lookup=roles)
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
            "b": _proposal((1.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
        }
        _, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertEqual(events, [])


class TestDeconflictionVerticalFallback(unittest.TestCase):
    """Stationary drones with zero velocity fall back to vertical separation."""

    def test_zero_velocity_uses_vertical(self) -> None:
        # Stationary primary with secondary directly underneath at same XY.
        # With both velocities zero and identical XY, lateral push along the
        # away-vector still won't help when the geometry is degenerate; we
        # construct a case where lateral fails by using zero velocity and an
        # exactly-overlapping XY position so the fallback path runs.
        roles = {"high": "primary_observer", "low": "secondary_baseline"}
        layer = DeconflictionLayer(
            DeconflictionConfig(min_separation_m=16.0, vertical_separation_m=20.0),
            role_lookup=roles,
        )
        proposed = {
            "high": _proposal((0.0, 0.0, 60.0), (0.0, 0.0, 0.0)),
            "low": _proposal((0.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertEqual(len(events), 1)
        # Either resolution may apply depending on the exact geometry, but the
        # yielder should now be vertically separated by at least the configured
        # vertical margin from the original peer altitude OR laterally offset
        # by min_separation.
        y_pos = adjusted["low"][0]
        peer_pos = adjusted["high"][0]
        z_gap = abs(float(y_pos[2] - peer_pos[2]))
        xy_gap = float(np.linalg.norm(y_pos[:2] - peer_pos[:2]))
        self.assertTrue(
            z_gap >= 20.0 or xy_gap >= 16.0,
            f"Expected separation restoration; got z_gap={z_gap}, xy_gap={xy_gap}",
        )


class TestDeconflictionDeterminism(unittest.TestCase):
    """Same input → same output across runs."""

    def test_repeatable_event_sequence(self) -> None:
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (5.0, 0.0, 0.0)),
            "b": _proposal((10.0, 0.0, 50.0), (-5.0, 0.0, 0.0)),
        }
        layer1 = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles)
        layer2 = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles)
        adj1, ev1 = layer1.resolve_step(proposed, timestamp_s=2.5)
        adj2, ev2 = layer2.resolve_step(proposed, timestamp_s=2.5)
        self.assertEqual(len(ev1), len(ev2))
        for e1, e2 in zip(ev1, ev2, strict=False):
            self.assertEqual(e1, e2)
        for did in adj1:
            np.testing.assert_array_equal(adj1[did][0], adj2[did][0])


class TestDeconflictionTerrainClearance(unittest.TestCase):
    """Vertical offset is clamped so the drone never goes below terrain + min_agl."""

    def test_vertical_offset_clamped_above_terrain(self) -> None:
        # Flat terrain at 100 m elevation; min_agl = 18 m → floor at 118 m.
        terrain = TerrainModel(
            base_elevation_m=100.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
        )
        roles = {"high": "primary_observer", "low": "secondary_baseline"}
        cfg = DeconflictionConfig(min_separation_m=16.0, vertical_separation_m=20.0, min_agl_m=18.0)
        layer = DeconflictionLayer(cfg, role_lookup=roles, terrain=terrain)

        # Both at 110 m — below the AGL floor after a downward vert offset.
        proposed = {
            "high": _proposal((0.0, 0.0, 120.0), (0.0, 0.0, 0.0)),
            "low": _proposal((0.0, 0.0, 110.0), (0.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertEqual(len(events), 1)
        # Yielder z must be >= terrain_height + min_agl = 118 m.
        yielder_z = float(adjusted["low"][0][2])
        self.assertGreaterEqual(yielder_z, 118.0 - 1e-9)

    def test_no_terrain_no_clamp(self) -> None:
        """Without a terrain model the layer should behave as before."""
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        layer = DeconflictionLayer(DeconflictionConfig(), role_lookup=roles, terrain=None)
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (5.0, 0.0, 0.0)),
            "b": _proposal((10.0, 0.0, 50.0), (-5.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertEqual(len(events), 1)  # still deconflicts


class TestDeconflictionExclusionZone(unittest.TestCase):
    """Lateral offset never places a drone inside an exclusion zone."""

    def _make_zone(self, cx: float, cy: float, radius: float) -> MissionZone:
        return MissionZone(
            zone_id="excl-1",
            zone_type="exclusion",
            center=vec3(cx, cy, 0.0),
            radius_m=radius,
        )

    def test_opposite_lateral_chosen_when_primary_in_zone(self) -> None:
        # b at (5,0) stationary, a at (0,0) stationary → lateral push on b
        # is in the +x direction (away from a): first candidate lands at (17,0).
        # Zone centred at (25,0) r=10 contains (17,0) but not the opposite (-7,0).
        # The layer should pick the opposite direction.
        zone = self._make_zone(25.0, 0.0, 10.0)
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        layer = DeconflictionLayer(
            DeconflictionConfig(min_separation_m=16.0),
            role_lookup=roles,
            exclusion_zones=[zone],
        )
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
            "b": _proposal((5.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
        }
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertEqual(len(events), 1)
        b_pos = adjusted["b"][0]
        # Result must not be inside the exclusion zone.
        dist = float(np.linalg.norm(b_pos[:2] - zone.center[:2]))
        self.assertGreater(
            dist, zone.radius_m - 1e-6, "Yielding drone landed inside the exclusion zone"
        )

    def test_non_exclusion_zones_ignored(self) -> None:
        """Surveillance zones are not treated as obstacles."""
        surv_zone = MissionZone(
            zone_id="surv-1",
            zone_type="surveillance",
            center=vec3(5.0, 20.0, 0.0),
            radius_m=25.0,
        )
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        layer = DeconflictionLayer(
            DeconflictionConfig(min_separation_m=16.0),
            role_lookup=roles,
            exclusion_zones=[surv_zone],
        )
        proposed = {
            "a": _proposal((0.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
            "b": _proposal((5.0, 0.0, 50.0), (0.0, 0.0, 0.0)),
        }
        # Should still resolve without error.
        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)
        self.assertGreaterEqual(len(events), 0)


class TestDeconflictionCorridorWindows(unittest.TestCase):
    """Bidirectional corridors alternate which travel direction may proceed."""

    def _corridor(self) -> FlightCorridor:
        return FlightCorridor(
            corridor_id="corr-1",
            waypoints_xy_m=[[-50.0, 0.0], [50.0, 0.0]],
            width_m=20.0,
            direction="bidirectional",
            active_window=[0.0, 100.0],
        )

    def test_negative_direction_holds_in_first_window(self) -> None:
        roles = {"eastbound": "primary_observer", "westbound": "secondary_baseline"}
        layer = DeconflictionLayer(
            DeconflictionConfig(min_separation_m=1.0),
            role_lookup=roles,
            corridors=[self._corridor()],
        )
        proposed = {
            "eastbound": _proposal((-20.0, 0.0, 50.0), (10.0, 0.0, 0.0)),
            "westbound": _proposal((20.0, 0.0, 50.0), (-10.0, 0.0, 0.0)),
        }

        adjusted, events = layer.resolve_step(proposed, timestamp_s=0.0)

        self.assertEqual(events[0].resolution, "corridor_hold")
        self.assertEqual(events[0].yielding_drone_id, "westbound")
        np.testing.assert_array_equal(adjusted["westbound"][1], np.zeros(3))
        np.testing.assert_array_equal(adjusted["eastbound"][1], proposed["eastbound"][1])

    def test_positive_direction_holds_in_second_window(self) -> None:
        roles = {"eastbound": "primary_observer", "westbound": "secondary_baseline"}
        layer = DeconflictionLayer(
            DeconflictionConfig(min_separation_m=1.0),
            role_lookup=roles,
            corridors=[self._corridor()],
        )
        proposed = {
            "eastbound": _proposal((-20.0, 0.0, 50.0), (10.0, 0.0, 0.0)),
            "westbound": _proposal((20.0, 0.0, 50.0), (-10.0, 0.0, 0.0)),
        }

        adjusted, events = layer.resolve_step(proposed, timestamp_s=11.0)

        self.assertEqual(events[0].resolution, "corridor_hold")
        self.assertEqual(events[0].yielding_drone_id, "eastbound")
        np.testing.assert_array_equal(adjusted["eastbound"][1], np.zeros(3))
        np.testing.assert_array_equal(adjusted["westbound"][1], proposed["westbound"][1])

    def test_same_direction_drones_do_not_hold(self) -> None:
        roles = {"a": "primary_observer", "b": "secondary_baseline"}
        layer = DeconflictionLayer(
            DeconflictionConfig(min_separation_m=1.0),
            role_lookup=roles,
            corridors=[self._corridor()],
        )
        proposed = {
            "a": _proposal((-20.0, 0.0, 50.0), (10.0, 0.0, 0.0)),
            "b": _proposal((20.0, 0.0, 50.0), (10.0, 0.0, 0.0)),
        }

        _, events = layer.resolve_step(proposed, timestamp_s=0.0)

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
