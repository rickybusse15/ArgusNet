"""WGS84, ECEF, and East-North-Up coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt

import numpy as np

# WGS84 ellipsoid constants
_WGS84_A = 6_378_137.0  # semi-major axis (m)
_WGS84_F = 1.0 / 298.257223563  # flattening
_WGS84_B = _WGS84_A * (1.0 - _WGS84_F)  # semi-minor axis (m)
_WGS84_E2 = 2.0 * _WGS84_F - _WGS84_F**2  # first eccentricity squared


@dataclass(frozen=True)
class ENUOrigin:
    """Reference point for local East-North-Up coordinates."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0


def _prime_vertical_radius(sin_lat: float) -> float:
    return _WGS84_A / sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """Convert geodetic WGS84 to Earth-Centred Earth-Fixed (ECEF) coordinates."""
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    sin_lat, cos_lat = sin(lat), cos(lat)
    sin_lon, cos_lon = sin(lon), cos(lon)
    n = _prime_vertical_radius(sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - _WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_enu(ecef: np.ndarray, origin: ENUOrigin) -> np.ndarray:
    """Convert ECEF coordinates to ENU relative to *origin*."""
    origin_ecef = wgs84_to_ecef(origin.latitude_deg, origin.longitude_deg, origin.altitude_m)
    dx, dy, dz = ecef - origin_ecef
    lat = radians(origin.latitude_deg)
    lon = radians(origin.longitude_deg)
    sin_lat, cos_lat = sin(lat), cos(lat)
    sin_lon, cos_lon = sin(lon), cos(lon)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return np.array([e, n, u], dtype=float)


def wgs84_to_enu(lat_deg: float, lon_deg: float, alt_m: float, origin: ENUOrigin) -> np.ndarray:
    """Convert geodetic WGS84 directly to local ENU coordinates."""
    return ecef_to_enu(wgs84_to_ecef(lat_deg, lon_deg, alt_m), origin)


def ecef_to_wgs84(ecef: np.ndarray) -> tuple[float, float, float]:
    """Convert ECEF back to geodetic WGS84 (lat_deg, lon_deg, alt_m).

    Uses Bowring's iterative method (converges in 2-3 iterations).
    """
    x, y, z = float(ecef[0]), float(ecef[1]), float(ecef[2])
    lon = atan2(y, x)
    p = sqrt(x * x + y * y)
    # initial estimate
    lat = atan2(z, p * (1.0 - _WGS84_E2))
    for _ in range(5):
        sin_lat = sin(lat)
        n = _prime_vertical_radius(sin_lat)
        lat = atan2(z + _WGS84_E2 * n * sin_lat, p)
    sin_lat = sin(lat)
    cos_lat = cos(lat)
    n = _prime_vertical_radius(sin_lat)
    alt = p / cos_lat - n if abs(cos_lat) > 1e-10 else abs(z) / abs(sin_lat) - n * (1.0 - _WGS84_E2)
    return np.degrees(lat), np.degrees(lon), alt


def enu_to_ecef(enu: np.ndarray, origin: ENUOrigin) -> np.ndarray:
    """Convert local ENU coordinates back to ECEF."""
    e, n, u = float(enu[0]), float(enu[1]), float(enu[2])
    lat = radians(origin.latitude_deg)
    lon = radians(origin.longitude_deg)
    sin_lat, cos_lat = sin(lat), cos(lat)
    sin_lon, cos_lon = sin(lon), cos(lon)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u
    origin_ecef = wgs84_to_ecef(origin.latitude_deg, origin.longitude_deg, origin.altitude_m)
    return origin_ecef + np.array([dx, dy, dz], dtype=float)


def enu_to_wgs84(enu: np.ndarray, origin: ENUOrigin) -> tuple[float, float, float]:
    """Convert local ENU coordinates back to geodetic WGS84."""
    return ecef_to_wgs84(enu_to_ecef(enu, origin))
