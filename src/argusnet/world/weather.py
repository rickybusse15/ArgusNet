"""Weather effects model for trajectory simulation.

Provides wind, atmospheric, precipitation, and cloud models that affect
sensor performance, visibility, and flight dynamics.  All physical units
follow SI conventions unless noted otherwise:

* Distances / altitudes: metres (m)
* Speeds: metres per second (m/s)
* Angles: radians (rad)
* Time: seconds (s)
* Pressure: hectopascals (hPa)
* Temperature: degrees Celsius (C)
* Precipitation rate: millimetres per hour (mm/h)
* Frequency: gigahertz (GHz)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindModel:
    """Constant-base plus sinusoidal-gust wind model.

    Parameters
    ----------
    base_speed_mps : float
        Steady-state wind speed (m/s).
    base_heading_rad : float
        Direction the wind blows *from*, measured clockwise from north
        (0 = from north, pi/2 = from east).  The resulting velocity vector
        points in the *downwind* direction.
    gust_amplitude_mps : float
        Peak gust speed above the base speed (m/s).
    gust_period_s : float
        Oscillation period of the sinusoidal gust (s).  Values <= 0 disable
        gusts.
    altitude_scaling : float
        Multiplicative factor applied to wind speed per 100 m of altitude
        above the surface.  A value of 0.3 means wind speed increases by
        30 % for every 100 m gained.
    """

    base_speed_mps: float = 0.0
    base_heading_rad: float = 0.0
    gust_amplitude_mps: float = 0.0
    gust_period_s: float = 10.0
    altitude_scaling: float = 0.0

    def wind_at(self, altitude_m: float, time_s: float) -> np.ndarray:
        """Return the 3-D wind vector ``[east, north, down]`` (m/s).

        The wind vector points in the direction the air mass is moving
        (downwind).  The vertical component is always zero in this model.

        Parameters
        ----------
        altitude_m : float
            Altitude above mean sea level (m).
        time_s : float
            Simulation clock time (s).
        """
        # Altitude scaling: speed grows linearly with altitude
        alt_factor = 1.0 + self.altitude_scaling * max(altitude_m, 0.0) / 100.0

        # Gust component (sinusoidal oscillation)
        if self.gust_period_s > 0.0 and self.gust_amplitude_mps > 0.0:
            gust = self.gust_amplitude_mps * math.sin(2.0 * math.pi * time_s / self.gust_period_s)
        else:
            gust = 0.0

        speed = max((self.base_speed_mps + gust) * alt_factor, 0.0)

        # Convert heading-from to downwind vector in ENU
        # heading_rad is clockwise from north (compass convention).
        # East component = sin(heading) (downwind, so same sign)
        # North component = cos(heading) (downwind, so same sign)
        # We want the *downwind* direction, which is the direction the wind
        # pushes objects.  "From north" means air moves southward, so we negate.
        east = -speed * math.sin(self.base_heading_rad)
        north = -speed * math.cos(self.base_heading_rad)
        return np.array([east, north, 0.0], dtype=float)


# ---------------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtmosphericConditions:
    """Atmospheric state affecting sensor performance.

    Parameters
    ----------
    visibility_m : float
        Meteorological visibility (m).  Standard clear-day value ~ 10 000 m.
    temperature_c : float
        Ambient temperature (degrees Celsius).
    relative_humidity : float
        Relative humidity in the range [0, 1].
    pressure_hpa : float
        Atmospheric pressure (hPa / mbar).
    """

    visibility_m: float = 10000.0
    temperature_c: float = 15.0
    relative_humidity: float = 0.5
    pressure_hpa: float = 1013.25

    def attenuation_factor(self, range_m: float, frequency_ghz: float = 0.0) -> float:
        """Signal power surviving propagation over *range_m* metres.

        Uses the Beer-Lambert law with an extinction coefficient derived from
        meteorological visibility (Koschmieder relation) and an optional
        frequency-dependent rain/humidity term.

        Returns a value in ``(0, 1]`` where 1 means no loss.

        Parameters
        ----------
        range_m : float
            One-way propagation distance (m).
        frequency_ghz : float, optional
            Carrier frequency (GHz).  When non-zero an additional
            humidity-dependent absorption term is added (simplified).
        """
        if range_m <= 0.0:
            return 1.0

        vis = max(self.visibility_m, 1.0)
        # Koschmieder: extinction coeff = 3.912 / V  (for 2% contrast threshold)
        sigma_ext = 3.912 / vis
        optical = math.exp(-sigma_ext * range_m)

        # Optional RF absorption (very simplified ITU model sketch)
        if frequency_ghz > 0.0:
            # Water-vapour specific attenuation grows with humidity & frequency
            rho_w = self.relative_humidity * 7.5  # g/m^3 (rough)
            gamma_w = 0.05 * rho_w * (frequency_ghz / 22.235) ** 2 / 1000.0  # dB/m
            rf_factor = 10.0 ** (-gamma_w * range_m / 10.0)
            return max(optical * rf_factor, 0.0)

        return max(optical, 0.0)

    def refraction_offset_rad(self, elevation_rad: float, range_m: float) -> float:
        """Angular offset caused by atmospheric refraction.

        A positive return value means the apparent elevation is *higher* than
        the geometric elevation.  Uses the standard atmosphere refraction
        approximation.

        Parameters
        ----------
        elevation_rad : float
            Geometric elevation angle above the horizon (rad).
        range_m : float
            Slant range to the target (m).
        """
        if range_m <= 0.0 or elevation_rad > math.pi / 2.0:
            return 0.0

        # Bennett's empirical formula (simplified, in radians)
        # Refraction ~ 1/(tan(el) + ... ) scaled by P/T
        el_deg = max(math.degrees(elevation_rad), 0.1)
        r_arcmin = 1.0 / math.tan(math.radians(el_deg + 7.31 / (el_deg + 4.4)))
        # Pressure/temperature correction
        correction = (self.pressure_hpa / 1010.0) * (283.0 / (273.0 + self.temperature_c))
        return math.radians(r_arcmin * correction / 60.0)


# ---------------------------------------------------------------------------
# Precipitation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecipitationModel:
    """Precipitation effects on visibility and sensor noise.

    Parameters
    ----------
    precip_type : str
        One of ``"none"``, ``"rain"``, ``"snow"``, ``"sleet"``, ``"fog"``.
    rate_mmph : float
        Precipitation rate (mm/h).  Ignored when *precip_type* is
        ``"none"`` or ``"fog"``.
    """

    precip_type: str = "none"
    rate_mmph: float = 0.0

    def __post_init__(self) -> None:
        valid = {"none", "rain", "snow", "sleet", "fog"}
        if self.precip_type not in valid:
            raise ValueError(
                f"precip_type must be one of {sorted(valid)}, got {self.precip_type!r}"
            )

    def visibility_reduction_factor(self) -> float:
        """Multiplicative factor on base meteorological visibility.

        Returns a value in ``(0, 1]`` where 1 means no reduction.  Uses
        empirical formulas relating precipitation rate to visibility.
        """
        if self.precip_type == "none":
            return 1.0
        if self.precip_type == "fog":
            # Fog is handled primarily through AtmosphericConditions.visibility_m
            # but we still apply a small penalty here.
            return 0.15
        rate = max(self.rate_mmph, 0.0)
        if rate <= 0.0:
            return 1.0
        if self.precip_type == "rain":
            # Marshall-Palmer empirical: V ~ 1800 / R^0.6  (metres)
            # Normalise to a 10 km clear-sky visibility.
            effective_vis = 1800.0 / max(rate**0.6, 0.01)
            return min(effective_vis / 10000.0, 1.0)
        if self.precip_type == "snow":
            # Snow reduces visibility more aggressively.
            effective_vis = 1200.0 / max(rate**0.75, 0.01)
            return min(effective_vis / 10000.0, 1.0)
        if self.precip_type == "sleet":
            effective_vis = 1500.0 / max(rate**0.65, 0.01)
            return min(effective_vis / 10000.0, 1.0)
        return 1.0  # pragma: no cover

    def sensor_noise_multiplier(self) -> float:
        """Multiplicative factor on sensor bearing-noise standard deviation.

        Always ``>= 1.0``.  Heavier precipitation causes more scatter and
        clutter, increasing bearing uncertainty.
        """
        if self.precip_type == "none":
            return 1.0
        if self.precip_type == "fog":
            return 1.6
        rate = max(self.rate_mmph, 0.0)
        if rate <= 0.0:
            return 1.0
        if self.precip_type == "rain":
            return 1.0 + 0.15 * rate**0.5
        if self.precip_type == "snow":
            return 1.0 + 0.25 * rate**0.5
        if self.precip_type == "sleet":
            return 1.0 + 0.20 * rate**0.5
        return 1.0  # pragma: no cover


# ---------------------------------------------------------------------------
# Cloud layers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloudLayer:
    """Simple cloud slab model.

    Parameters
    ----------
    base_altitude_m : float
        Altitude of the cloud base (m above sea level).
    top_altitude_m : float
        Altitude of the cloud top (m above sea level).
    coverage : float
        Fraction of sky covered, ``[0, 1]``.
    """

    base_altitude_m: float = 3000.0
    top_altitude_m: float = 4000.0
    coverage: float = 0.0

    def obscures_los(self, observer_alt_m: float, target_alt_m: float) -> bool:
        """Return ``True`` if the cloud layer blocks line of sight.

        The check is purely altitude-based: the LOS is blocked when the
        observer and target are on opposite sides of the cloud slab and
        the coverage is 100 %.  Partial coverage is treated probabilistically
        elsewhere; here we report a hard block only when *coverage* is 1.0.

        Parameters
        ----------
        observer_alt_m : float
            Observer altitude (m ASL).
        target_alt_m : float
            Target altitude (m ASL).
        """
        if self.coverage < 1.0:
            return False
        low = min(observer_alt_m, target_alt_m)
        high = max(observer_alt_m, target_alt_m)
        # Blocked if the slab sits between observer and target
        return low < self.base_altitude_m and high > self.top_altitude_m


# ---------------------------------------------------------------------------
# Composite weather model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeatherModel:
    """Complete weather state for a simulation time-step.

    Composes wind, atmosphere, precipitation, and cloud layers into a
    single query interface used by the simulation engine.
    """

    wind: WindModel = WindModel()
    atmosphere: AtmosphericConditions = AtmosphericConditions()
    precipitation: PrecipitationModel = PrecipitationModel()
    cloud_layers: tuple[CloudLayer, ...] = ()

    def visibility_at_range(self, range_m: float) -> float:
        """Effective visibility probability at *range_m* metres.

        Combines atmospheric attenuation with precipitation-based visibility
        reduction.  Returns a value in ``[0, 1]`` representing the probability
        that a target at *range_m* is detectable.

        Parameters
        ----------
        range_m : float
            Slant range to the target (m).
        """
        precip_factor = self.precipitation.visibility_reduction_factor()
        effective_vis = max(self.atmosphere.visibility_m * precip_factor, 1.0)
        # Use Koschmieder with the effective visibility
        sigma = 3.912 / effective_vis
        return max(math.exp(-sigma * max(range_m, 0.0)), 0.0)

    def bearing_noise_scale(self) -> float:
        """Multiplicative factor on bearing-observation noise.

        Always ``>= 1.0``.  Combines precipitation noise with a small
        humidity-driven penalty.

        Returns
        -------
        float
        """
        precip_noise = self.precipitation.sensor_noise_multiplier()
        # Additional mild penalty for very humid air (condensation on optics)
        humidity_noise = 1.0 + 0.3 * max(self.atmosphere.relative_humidity - 0.8, 0.0)
        return max(precip_noise * humidity_noise, 1.0)

    def flight_speed_penalty(self, altitude_m: float) -> float:
        """Speed-reduction factor due to weather at *altitude_m*.

        Returns a value in ``(0, 1]`` where 1 means no penalty.  The penalty
        is driven primarily by wind speed relative to a nominal drone cruise
        speed (15 m/s) and a precipitation drag term.

        Parameters
        ----------
        altitude_m : float
            Flight altitude (m ASL).
        """
        nominal_cruise_mps = 15.0
        wind_vec = self.wind.wind_at(altitude_m, 0.0)
        wind_speed = float(np.linalg.norm(wind_vec))
        wind_penalty = max(1.0 - 0.5 * (wind_speed / nominal_cruise_mps), 0.1)

        # Precipitation drag
        precip_drag = 1.0
        if self.precipitation.precip_type == "rain":
            precip_drag = max(1.0 - 0.02 * self.precipitation.rate_mmph, 0.5)
        elif self.precipitation.precip_type == "snow":
            precip_drag = max(1.0 - 0.04 * self.precipitation.rate_mmph, 0.4)
        elif self.precipitation.precip_type == "sleet":
            precip_drag = max(1.0 - 0.03 * self.precipitation.rate_mmph, 0.45)
        elif self.precipitation.precip_type == "fog":
            precip_drag = 0.9

        return max(wind_penalty * precip_drag, 0.05)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

WEATHER_PRESETS: dict[str, WeatherModel] = {
    "clear": WeatherModel(
        wind=WindModel(
            base_speed_mps=3.0,
            base_heading_rad=math.radians(180),
            gust_amplitude_mps=1.0,
            gust_period_s=20.0,
            altitude_scaling=0.2,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=15000.0,
            temperature_c=20.0,
            relative_humidity=0.35,
            pressure_hpa=1018.0,
        ),
        precipitation=PrecipitationModel(precip_type="none", rate_mmph=0.0),
        cloud_layers=(),
    ),
    "overcast": WeatherModel(
        wind=WindModel(
            base_speed_mps=5.0,
            base_heading_rad=math.radians(240),
            gust_amplitude_mps=2.0,
            gust_period_s=15.0,
            altitude_scaling=0.25,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=8000.0,
            temperature_c=12.0,
            relative_humidity=0.70,
            pressure_hpa=1008.0,
        ),
        precipitation=PrecipitationModel(precip_type="none", rate_mmph=0.0),
        cloud_layers=(CloudLayer(base_altitude_m=1500.0, top_altitude_m=3000.0, coverage=0.85),),
    ),
    "light_rain": WeatherModel(
        wind=WindModel(
            base_speed_mps=6.0,
            base_heading_rad=math.radians(210),
            gust_amplitude_mps=3.0,
            gust_period_s=12.0,
            altitude_scaling=0.30,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=6000.0,
            temperature_c=10.0,
            relative_humidity=0.85,
            pressure_hpa=1002.0,
        ),
        precipitation=PrecipitationModel(precip_type="rain", rate_mmph=2.5),
        cloud_layers=(CloudLayer(base_altitude_m=800.0, top_altitude_m=2500.0, coverage=0.95),),
    ),
    "heavy_rain": WeatherModel(
        wind=WindModel(
            base_speed_mps=12.0,
            base_heading_rad=math.radians(200),
            gust_amplitude_mps=7.0,
            gust_period_s=8.0,
            altitude_scaling=0.35,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=2000.0,
            temperature_c=8.0,
            relative_humidity=0.95,
            pressure_hpa=995.0,
        ),
        precipitation=PrecipitationModel(precip_type="rain", rate_mmph=25.0),
        cloud_layers=(CloudLayer(base_altitude_m=400.0, top_altitude_m=2000.0, coverage=1.0),),
    ),
    "fog": WeatherModel(
        wind=WindModel(
            base_speed_mps=1.5,
            base_heading_rad=math.radians(90),
            gust_amplitude_mps=0.5,
            gust_period_s=30.0,
            altitude_scaling=0.10,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=200.0,
            temperature_c=5.0,
            relative_humidity=0.98,
            pressure_hpa=1015.0,
        ),
        precipitation=PrecipitationModel(precip_type="fog", rate_mmph=0.0),
        cloud_layers=(CloudLayer(base_altitude_m=0.0, top_altitude_m=300.0, coverage=1.0),),
    ),
    "snow": WeatherModel(
        wind=WindModel(
            base_speed_mps=4.0,
            base_heading_rad=math.radians(0),
            gust_amplitude_mps=2.0,
            gust_period_s=18.0,
            altitude_scaling=0.25,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=3000.0,
            temperature_c=-5.0,
            relative_humidity=0.80,
            pressure_hpa=1005.0,
        ),
        precipitation=PrecipitationModel(precip_type="snow", rate_mmph=5.0),
        cloud_layers=(CloudLayer(base_altitude_m=600.0, top_altitude_m=3500.0, coverage=0.90),),
    ),
    "storm": WeatherModel(
        wind=WindModel(
            base_speed_mps=20.0,
            base_heading_rad=math.radians(225),
            gust_amplitude_mps=12.0,
            gust_period_s=5.0,
            altitude_scaling=0.40,
        ),
        atmosphere=AtmosphericConditions(
            visibility_m=1000.0,
            temperature_c=6.0,
            relative_humidity=0.97,
            pressure_hpa=985.0,
        ),
        precipitation=PrecipitationModel(precip_type="rain", rate_mmph=50.0),
        cloud_layers=(
            CloudLayer(base_altitude_m=200.0, top_altitude_m=1200.0, coverage=1.0),
            CloudLayer(base_altitude_m=1500.0, top_altitude_m=8000.0, coverage=1.0),
        ),
    ),
}

KNOWN_WEATHER_PRESETS = frozenset(WEATHER_PRESETS)


def weather_from_preset(name: str) -> WeatherModel:
    """Load a weather preset by name.

    Parameters
    ----------
    name : str
        One of the keys in :data:`WEATHER_PRESETS`.

    Returns
    -------
    WeatherModel

    Raises
    ------
    ValueError
        If *name* is not a recognised preset.
    """
    try:
        return WEATHER_PRESETS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown weather preset {name!r}. Available: {sorted(WEATHER_PRESETS)}"
        ) from error


__all__ = [
    "AtmosphericConditions",
    "CloudLayer",
    "KNOWN_WEATHER_PRESETS",
    "PrecipitationModel",
    "WEATHER_PRESETS",
    "WeatherModel",
    "WindModel",
    "weather_from_preset",
]
