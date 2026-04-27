"""Tests for WGS84 / ECEF / ENU coordinate transforms."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.frames import (
    ENUOrigin,
    ecef_to_enu,
    ecef_to_wgs84,
    enu_to_ecef,
    enu_to_wgs84,
    wgs84_to_ecef,
    wgs84_to_enu,
)


class TestWGS84ToECEF(unittest.TestCase):
    def test_equator_prime_meridian(self):
        ecef = wgs84_to_ecef(0.0, 0.0, 0.0)
        self.assertAlmostEqual(ecef[0], 6_378_137.0, places=1)
        self.assertAlmostEqual(ecef[1], 0.0, places=1)
        self.assertAlmostEqual(ecef[2], 0.0, places=1)

    def test_north_pole(self):
        ecef = wgs84_to_ecef(90.0, 0.0, 0.0)
        self.assertAlmostEqual(ecef[0], 0.0, places=1)
        self.assertAlmostEqual(ecef[1], 0.0, places=1)
        self.assertGreater(ecef[2], 6_350_000.0)

    def test_south_pole(self):
        ecef = wgs84_to_ecef(-90.0, 0.0, 0.0)
        self.assertAlmostEqual(ecef[0], 0.0, places=1)
        self.assertAlmostEqual(ecef[1], 0.0, places=1)
        self.assertLess(ecef[2], -6_350_000.0)

    def test_altitude_increases_radius(self):
        ecef_ground = wgs84_to_ecef(45.0, 10.0, 0.0)
        ecef_high = wgs84_to_ecef(45.0, 10.0, 1000.0)
        self.assertGreater(np.linalg.norm(ecef_high), np.linalg.norm(ecef_ground))


class TestECEFRoundTrip(unittest.TestCase):
    def _assert_round_trip(self, lat, lon, alt, tol_m=0.001):
        ecef = wgs84_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_wgs84(ecef)
        self.assertAlmostEqual(lat, lat2, places=8, msg="latitude mismatch")
        self.assertAlmostEqual(lon, lon2, places=8, msg="longitude mismatch")
        self.assertAlmostEqual(alt, alt2, delta=tol_m, msg="altitude mismatch")

    def test_zurich(self):
        self._assert_round_trip(47.3769, 8.5417, 408.0)

    def test_equator(self):
        self._assert_round_trip(0.0, 0.0, 0.0)

    def test_high_altitude(self):
        self._assert_round_trip(34.0, -118.0, 10000.0)

    def test_south_hemisphere(self):
        self._assert_round_trip(-33.8688, 151.2093, 58.0)

    def test_date_line(self):
        self._assert_round_trip(0.0, 179.999, 0.0)
        self._assert_round_trip(0.0, -179.999, 0.0)

    def test_north_pole(self):
        self._assert_round_trip(90.0, 0.0, 0.0, tol_m=0.01)

    def test_south_pole(self):
        self._assert_round_trip(-90.0, 0.0, 0.0, tol_m=0.01)


class TestENURoundTrip(unittest.TestCase):
    def test_origin_maps_to_zero(self):
        origin = ENUOrigin(47.3769, 8.5417, 408.0)
        enu = wgs84_to_enu(47.3769, 8.5417, 408.0, origin)
        np.testing.assert_allclose(enu, [0.0, 0.0, 0.0], atol=1e-6)

    def test_round_trip_nearby_point(self):
        origin = ENUOrigin(47.3769, 8.5417, 408.0)
        lat, lon, alt = 47.3780, 8.5430, 420.0
        enu = wgs84_to_enu(lat, lon, alt, origin)
        lat2, lon2, alt2 = enu_to_wgs84(enu, origin)
        self.assertAlmostEqual(lat, lat2, places=7)
        self.assertAlmostEqual(lon, lon2, places=7)
        self.assertAlmostEqual(alt, alt2, delta=0.001)

    def test_east_direction(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        enu = wgs84_to_enu(0.0, 0.001, 0.0, origin)
        self.assertGreater(enu[0], 0.0)  # east positive
        self.assertAlmostEqual(enu[1], 0.0, delta=1.0)  # north ~0
        self.assertAlmostEqual(enu[2], 0.0, delta=1.0)  # up ~0

    def test_north_direction(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        enu = wgs84_to_enu(0.001, 0.0, 0.0, origin)
        self.assertAlmostEqual(enu[0], 0.0, delta=1.0)  # east ~0
        self.assertGreater(enu[1], 0.0)  # north positive

    def test_up_direction(self):
        origin = ENUOrigin(47.0, 8.0, 0.0)
        enu = wgs84_to_enu(47.0, 8.0, 100.0, origin)
        self.assertAlmostEqual(enu[0], 0.0, delta=0.01)
        self.assertAlmostEqual(enu[1], 0.0, delta=0.01)
        self.assertAlmostEqual(enu[2], 100.0, delta=0.01)

    def test_ecef_enu_round_trip(self):
        origin = ENUOrigin(47.3769, 8.5417, 408.0)
        enu_in = np.array([100.0, 200.0, 50.0])
        ecef = enu_to_ecef(enu_in, origin)
        enu_out = ecef_to_enu(ecef, origin)
        np.testing.assert_allclose(enu_in, enu_out, atol=1e-6)

    def test_large_offset(self):
        origin = ENUOrigin(34.0522, -118.2437, 71.0)  # Los Angeles
        lat, lon, alt = 34.0622, -118.2337, 171.0  # ~1.4km away
        enu = wgs84_to_enu(lat, lon, alt, origin)
        lat2, lon2, alt2 = enu_to_wgs84(enu, origin)
        self.assertAlmostEqual(lat, lat2, places=6)
        self.assertAlmostEqual(lon, lon2, places=6)
        self.assertAlmostEqual(alt, alt2, delta=0.01)


class TestENUOriginDataclass(unittest.TestCase):
    def test_frozen(self):
        origin = ENUOrigin(47.0, 8.0, 400.0)
        with self.assertRaises(AttributeError):
            origin.latitude_deg = 48.0

    def test_default_altitude(self):
        origin = ENUOrigin(47.0, 8.0)
        self.assertEqual(origin.altitude_m, 0.0)


if __name__ == "__main__":
    unittest.main()
