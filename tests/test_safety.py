"""Tests for safety-critical subsystems: bounds clamping, terrain clearance, deconfliction."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.planning.deconfliction import DeconflictionConfig, DeconflictionLayer
from argusnet.simulation.sim import clamp_to_bounds, orbital_path
from argusnet.world.terrain import TerrainModel


class ClampToBoundsTest(unittest.TestCase):
    """Verify the oversized-margin guard introduced to fix the inverted-range bug."""

    _bounds = {"x_min_m": 0.0, "x_max_m": 10.0, "y_min_m": 0.0, "y_max_m": 10.0}

    def test_oversized_margin_collapses_to_midpoint(self) -> None:
        # margin=8 > span/2=5 → both axes must collapse to midpoint (5.0)
        result = clamp_to_bounds([0.0, 0.0], self._bounds, margin_m=8.0)
        self.assertAlmostEqual(float(result[0]), 5.0)
        self.assertAlmostEqual(float(result[1]), 5.0)

    def test_point_outside_bounds_is_clamped_in(self) -> None:
        bounds = {"x_min_m": 0.0, "x_max_m": 100.0, "y_min_m": 0.0, "y_max_m": 100.0}
        result = clamp_to_bounds([200.0, -50.0], bounds, margin_m=0.0)
        self.assertAlmostEqual(float(result[0]), 100.0)
        self.assertAlmostEqual(float(result[1]), 0.0)

    def test_margin_respected_for_normal_span(self) -> None:
        bounds = {"x_min_m": 0.0, "x_max_m": 100.0, "y_min_m": 0.0, "y_max_m": 100.0}
        result = clamp_to_bounds([0.0, 0.0], bounds, margin_m=10.0)
        self.assertGreaterEqual(float(result[0]), 10.0)
        self.assertGreaterEqual(float(result[1]), 10.0)


class TerrainClearanceTest(unittest.TestCase):
    """Verify that trajectory generators enforce min AGL regardless of requested altitude."""

    def _flat_terrain(self, ground_plane_m: float = 0.0) -> TerrainModel:
        return TerrainModel(
            ground_plane_m=ground_plane_m,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.0,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )

    def test_orbital_path_min_agl_enforced_despite_very_low_base_altitude(self) -> None:
        terrain = self._flat_terrain(ground_plane_m=50.0)
        trajectory = orbital_path(
            center_xy=np.array([0.0, 0.0], dtype=float),
            radii_xy=np.array([30.0, 30.0], dtype=float),
            base_agl_m=-200.0,
            vertical_amplitude_m=0.0,
            omega=0.0,
            phase=0.0,
            terrain=terrain,
            min_agl_m=15.0,
        )
        for t in np.linspace(0.0, 60.0, 24):
            position, _ = trajectory(float(t))
            self.assertGreaterEqual(
                position[2],
                terrain.ground_plane_m + 15.0,
                msg=f"Altitude {position[2]:.1f} m violated terrain floor at t={t:.1f}",
            )

    def test_terrain_clamp_altitude_raises_point_below_floor(self) -> None:
        terrain = self._flat_terrain(ground_plane_m=100.0)
        clamped_z = terrain.clamp_altitude([0.0, 0.0], z_m=80.0, min_agl_m=10.0)
        self.assertGreaterEqual(clamped_z, terrain.ground_plane_m + 10.0)

    def test_orbital_path_min_agl_on_sloped_terrain(self) -> None:
        sloped = TerrainModel(
            ground_plane_m=0.0,
            base_elevation_m=0.0,
            slope_x_m_per_m=0.2,
            slope_y_m_per_m=0.0,
            wave_amplitude_m=0.0,
            ridge_amplitude_m=0.0,
            basin_depth_m=0.0,
        )
        trajectory = orbital_path(
            center_xy=np.array([0.0, 0.0], dtype=float),
            radii_xy=np.array([100.0, 100.0], dtype=float),
            base_agl_m=0.0,
            vertical_amplitude_m=0.0,
            omega=0.0,
            phase=0.0,
            terrain=sloped,
            min_agl_m=8.0,
        )
        for t in np.linspace(0.0, 30.0, 12):
            position, _ = trajectory(float(t))
            terrain_height = sloped.height_at(float(position[0]), float(position[1]))
            self.assertGreaterEqual(
                position[2],
                terrain_height + 8.0 - 1e-6,
                msg=f"Altitude {position[2]:.1f} m < terrain+8 at t={t:.1f}",
            )


class DeconflictionSafetyTest(unittest.TestCase):
    """Verify that the deconfliction layer never allows minimum separation to be violated."""

    def test_separation_maintained_over_multiple_steps_under_constant_pressure(self) -> None:
        cfg = DeconflictionConfig(min_separation_m=20.0, look_ahead_s=1.0)
        roles = {"drone-a": "primary_observer", "drone-b": "secondary_baseline"}
        layer = DeconflictionLayer(cfg, role_lookup=roles)

        # Two drones start 8 m apart (well below min sep) and converge at 5 m/s.
        pos_a = np.array([0.0, 0.0, 50.0], dtype=float)
        vel_a = np.array([3.0, 0.0, 0.0], dtype=float)
        pos_b = np.array([8.0, 0.0, 50.0], dtype=float)
        vel_b = np.array([-3.0, 0.0, 0.0], dtype=float)

        for step in range(10):
            proposed = {
                "drone-a": (pos_a.copy(), vel_a.copy()),
                "drone-b": (pos_b.copy(), vel_b.copy()),
            }
            adjusted, _ = layer.resolve_step(proposed, timestamp_s=float(step) * 0.5)
            sep = float(np.linalg.norm(adjusted["drone-a"][0] - adjusted["drone-b"][0]))
            self.assertGreaterEqual(
                sep,
                cfg.min_separation_m,
                msg=f"Separation {sep:.2f} m violated min {cfg.min_separation_m} m at step {step}",
            )
            pos_a = adjusted["drone-a"][0]
            pos_b = adjusted["drone-b"][0]

    def test_three_drone_pileup_all_pairwise_separations_maintained(self) -> None:
        cfg = DeconflictionConfig(min_separation_m=15.0)
        roles = {
            "d0": "primary_observer",
            "d1": "secondary_baseline",
            "d2": "secondary_baseline",
        }
        layer = DeconflictionLayer(cfg, role_lookup=roles)

        # All three drones at the same point — worst case pileup.
        proposed = {
            "d0": (np.array([0.0, 0.0, 50.0], dtype=float), np.zeros(3, dtype=float)),
            "d1": (np.array([2.0, 0.0, 50.0], dtype=float), np.zeros(3, dtype=float)),
            "d2": (np.array([-2.0, 0.0, 50.0], dtype=float), np.zeros(3, dtype=float)),
        }
        adjusted, _ = layer.resolve_step(proposed, timestamp_s=0.0)

        ids = sorted(adjusted.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sep = float(np.linalg.norm(adjusted[ids[i]][0] - adjusted[ids[j]][0]))
                self.assertGreaterEqual(
                    sep,
                    cfg.min_separation_m,
                    msg=f"Pair ({ids[i]},{ids[j]}) separation {sep:.2f} m < min {cfg.min_separation_m} m",
                )


if __name__ == "__main__":
    unittest.main()
