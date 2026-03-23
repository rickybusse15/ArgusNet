"""Backward-compatibility shim — imports from argusnet.world.weather."""
from argusnet.world.weather import *  # noqa: F401, F403
from argusnet.world.weather import (
    KNOWN_WEATHER_PRESETS,
    WEATHER_PRESETS,
    AtmosphericConditions,
    CloudLayer,
    PrecipitationModel,
    WeatherModel,
    WindModel,
    weather_from_preset,
)
