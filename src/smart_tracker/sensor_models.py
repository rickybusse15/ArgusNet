"""Backward-compatibility shim — imports from argusnet.sensing.models.noise."""
from argusnet.sensing.models.noise import *  # noqa: F401, F403
from argusnet.sensing.models.noise import (
    SENSOR_ERROR_PRESETS,
    SensorBiasDrift,
    SensorErrorConfig,
    SensorModel,
    apply_bias_to_direction,
    atmospheric_attenuation,
    detection_probability,
    generate_false_alarms,
    range_dependent_bearing_noise,
    sensor_error_config_from_preset,
    snr_at_range,
)
