"""Sensor error models for realistic observation synthesis.

Provides range-dependent noise, atmospheric attenuation, sensor bias drift,
false alarm (clutter) generation, and missed detection probability.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from argusnet.core.types import BearingObservation, vec3


# ---------------------------------------------------------------------------
# Sensor error configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorErrorConfig:
    """Complete sensor error model configuration.

    Controls how bearing noise, detection probability, false alarms, and
    bias drift behave for a sensor node during simulation.
    """

    # --- Range-dependent noise ---
    base_bearing_std_rad: float = 0.015
    """Bearing noise std at zero range (radians)."""

    range_noise_exponent: float = 1.5
    """Exponent for range-dependent noise growth: σ ∝ (R/R_ref)^exponent."""

    range_noise_reference_m: float = 1000.0
    """Reference range (m) at which noise equals base_bearing_std_rad."""

    max_bearing_std_rad: float = 0.15
    """Hard cap on bearing noise std (radians) to prevent degenerate observations."""

    # --- Atmospheric attenuation ---
    atmospheric_attenuation_coeff: float = 0.0001
    """Beer-Lambert extinction coefficient (1/m). Default ~clear air at visible wavelengths."""

    # --- Missed detection probability ---
    min_detection_probability: float = 0.1
    """Floor on P_d — even at max range, there's a small chance of detection."""

    detection_range_knee_m: float = 800.0
    """Range at which P_d starts to drop from 1.0. Below this, P_d ≈ 1.0."""

    detection_range_falloff_m: float = 400.0
    """Characteristic falloff distance: P_d halves every falloff_m past the knee."""

    # --- False alarms / clutter ---
    false_alarm_rate_per_scan: float = 0.0
    """Expected number of false alarms per sensor per scan (Poisson rate).
    Set to 0.0 to disable clutter. Typical: 0.01 - 0.5."""

    clutter_bearing_std_rad: float = 0.3
    """Bearing noise std for clutter (false alarm) observations (radians)."""

    clutter_min_range_m: float = 50.0
    """Minimum apparent range for clutter returns (m)."""

    clutter_max_range_m: float = 2000.0
    """Maximum apparent range for clutter returns (m)."""

    # --- Bias drift ---
    bias_drift_rate_rad_per_s: float = 0.0
    """Random walk std per second for sensor bearing bias drift (rad/√s).
    Set to 0.0 to disable bias drift. Typical: 1e-5 to 1e-4."""

    bias_drift_max_rad: float = 0.01
    """Maximum absolute bias drift before clamping (radians)."""

    # --- SNR model ---
    snr_reference_db: float = 30.0
    """SNR at reference range (dB)."""

    snr_range_exponent: float = 2.0
    """SNR drops as (R_ref/R)^exponent. 2.0 = inverse-square law."""


@dataclass(frozen=True)
class SensorErrorPreset:
    """Named sensor error preset."""
    name: str
    config: SensorErrorConfig


SENSOR_ERROR_PRESETS = {
    "ideal": SensorErrorConfig(
        base_bearing_std_rad=0.005,
        range_noise_exponent=0.5,
        atmospheric_attenuation_coeff=0.0,
        false_alarm_rate_per_scan=0.0,
        bias_drift_rate_rad_per_s=0.0,
        min_detection_probability=0.95,
    ),
    "baseline": SensorErrorConfig(),
    "degraded": SensorErrorConfig(
        base_bearing_std_rad=0.025,
        range_noise_exponent=2.0,
        atmospheric_attenuation_coeff=0.0003,
        false_alarm_rate_per_scan=0.1,
        bias_drift_rate_rad_per_s=5e-5,
        detection_range_knee_m=500.0,
        detection_range_falloff_m=300.0,
    ),
    "noisy": SensorErrorConfig(
        base_bearing_std_rad=0.04,
        range_noise_exponent=2.0,
        atmospheric_attenuation_coeff=0.0005,
        false_alarm_rate_per_scan=0.3,
        bias_drift_rate_rad_per_s=1e-4,
        detection_range_knee_m=400.0,
        detection_range_falloff_m=200.0,
        min_detection_probability=0.05,
    ),
}


def sensor_error_config_from_preset(name: str) -> SensorErrorConfig:
    """Load a sensor error preset by name.

    Raises ValueError if the preset name is unknown.
    """
    if name not in SENSOR_ERROR_PRESETS:
        raise ValueError(
            f"Unknown sensor error preset {name!r}. "
            f"Available: {sorted(SENSOR_ERROR_PRESETS)}"
        )
    return SENSOR_ERROR_PRESETS[name]


# ---------------------------------------------------------------------------
# Sensor bias drift tracker (stateful, per-node)
# ---------------------------------------------------------------------------

class SensorBiasDrift:
    """Tracks per-node bearing bias drift as a bounded random walk.

    Bias evolves as: bias[t+dt] = bias[t] + N(0, drift_rate * √dt)
    clamped to [-max_bias, +max_bias].
    """

    def __init__(
        self,
        drift_rate_rad_per_s: float = 0.0,
        max_bias_rad: float = 0.01,
        seed: int = 0,
    ) -> None:
        self._drift_rate = drift_rate_rad_per_s
        self._max_bias = max_bias_rad
        self._rng = np.random.default_rng(seed)
        self._current_bias: float = 0.0
        self._last_time: Optional[float] = None

    @property
    def current_bias_rad(self) -> float:
        return self._current_bias

    def step(self, timestamp_s: float) -> float:
        """Advance the bias drift to *timestamp_s* and return the current bias."""
        if self._drift_rate <= 0.0:
            return 0.0
        if self._last_time is not None:
            dt = max(timestamp_s - self._last_time, 0.0)
            if dt > 0.0:
                increment = self._rng.normal(0.0, self._drift_rate * math.sqrt(dt))
                self._current_bias += increment
                self._current_bias = float(
                    np.clip(self._current_bias, -self._max_bias, self._max_bias)
                )
        self._last_time = timestamp_s
        return self._current_bias

    def reset(self) -> None:
        self._current_bias = 0.0
        self._last_time = None


# ---------------------------------------------------------------------------
# Core sensor model functions
# ---------------------------------------------------------------------------

def range_dependent_bearing_noise(
    base_std_rad: float,
    range_m: float,
    config: SensorErrorConfig,
) -> float:
    """Compute bearing noise std as a function of range.

    σ(R) = base_std * (R / R_ref)^exponent, clamped to [base_std, max_std].
    """
    if range_m <= 0.0 or config.range_noise_reference_m <= 0.0:
        return base_std_rad
    ratio = range_m / config.range_noise_reference_m
    scaled = base_std_rad * (ratio ** config.range_noise_exponent)
    return float(np.clip(scaled, base_std_rad, config.max_bearing_std_rad))


def atmospheric_attenuation(
    range_m: float,
    config: SensorErrorConfig,
) -> float:
    """Beer-Lambert atmospheric transmittance over *range_m*.

    Returns a factor in [0, 1] where 1.0 = no attenuation.
    """
    if config.atmospheric_attenuation_coeff <= 0.0 or range_m <= 0.0:
        return 1.0
    return float(math.exp(-config.atmospheric_attenuation_coeff * range_m))


def snr_at_range(
    range_m: float,
    config: SensorErrorConfig,
) -> float:
    """Compute SNR in dB at *range_m* using inverse-power law.

    SNR(R) = SNR_ref - exponent * 10 * log10(R / R_ref)
    """
    if range_m <= 0.0 or config.range_noise_reference_m <= 0.0:
        return config.snr_reference_db
    ratio = range_m / config.range_noise_reference_m
    if ratio < 1e-12:
        return config.snr_reference_db
    return config.snr_reference_db - config.snr_range_exponent * 10.0 * math.log10(ratio)


def detection_probability(
    range_m: float,
    config: SensorErrorConfig,
    atmospheric_transmittance: float = 1.0,
) -> float:
    """Compute probability of detection at *range_m*.

    P_d = max(P_min, transmittance * sigmoid_falloff(R))
    """
    if range_m <= config.detection_range_knee_m:
        base_pd = 1.0
    elif config.detection_range_falloff_m <= 0.0:
        base_pd = config.min_detection_probability
    else:
        excess = range_m - config.detection_range_knee_m
        # Exponential falloff: P_d = exp(-ln(2) * excess / falloff)
        base_pd = math.exp(-math.log(2.0) * excess / config.detection_range_falloff_m)

    pd = base_pd * atmospheric_transmittance
    return float(np.clip(pd, config.min_detection_probability, 1.0))


def generate_false_alarms(
    rng: np.random.Generator,
    node_position: np.ndarray,
    config: SensorErrorConfig,
    timestamp_s: float,
    node_id: str,
) -> List[BearingObservation]:
    """Generate Poisson-distributed false alarm (clutter) observations.

    Returns a list of BearingObservation with target_id="clutter".
    """
    if config.false_alarm_rate_per_scan <= 0.0:
        return []

    count = rng.poisson(config.false_alarm_rate_per_scan)
    if count == 0:
        return []

    alarms: List[BearingObservation] = []
    for _ in range(count):
        # Random direction (uniform on unit sphere)
        direction = rng.normal(size=3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            continue
        direction = direction / norm

        # Add noise to make it realistic
        noise = rng.normal(0.0, config.clutter_bearing_std_rad, size=3)
        noisy_dir = direction + noise
        noisy_norm = float(np.linalg.norm(noisy_dir))
        if noisy_norm < 1e-12:
            continue
        noisy_dir = noisy_dir / noisy_norm

        alarms.append(
            BearingObservation(
                node_id=node_id,
                target_id="clutter",
                origin=node_position.copy(),
                direction=noisy_dir,
                bearing_std_rad=config.clutter_bearing_std_rad,
                timestamp_s=timestamp_s,
                confidence=float(rng.uniform(0.05, 0.4)),
            )
        )

    return alarms


def apply_bias_to_direction(
    direction: np.ndarray,
    bias_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rotate a direction vector by a bearing bias angle.

    The bias is applied as a rotation about a random axis perpendicular
    to the direction vector.
    """
    if abs(bias_rad) < 1e-12:
        return direction

    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-12:
        return direction
    unit_dir = direction / dir_norm

    # Find a perpendicular axis
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(unit_dir, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    perp = np.cross(unit_dir, arbitrary)
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-12:
        return direction
    perp = perp / perp_norm

    # Rodrigues rotation of unit_dir around perp by bias_rad
    cos_b = math.cos(bias_rad)
    sin_b = math.sin(bias_rad)
    rotated = (
        unit_dir * cos_b
        + np.cross(perp, unit_dir) * sin_b
        + perp * np.dot(perp, unit_dir) * (1.0 - cos_b)
    )
    rot_norm = float(np.linalg.norm(rotated))
    if rot_norm < 1e-12:
        return direction
    return rotated / rot_norm * dir_norm


# ---------------------------------------------------------------------------
# Composite sensor model: applies all effects to a single observation
# ---------------------------------------------------------------------------

@dataclass
class SensorModel:
    """Stateful sensor model that applies all error effects.

    Create one per sensor node, call `process_observation()` for each
    potential observation.
    """

    config: SensorErrorConfig = field(default_factory=SensorErrorConfig)
    bias_drift: Optional[SensorBiasDrift] = None
    _initialized: bool = field(default=False, init=False, repr=False)

    def initialize(self, seed: int = 0) -> None:
        """Initialize internal state (call once before simulation)."""
        if self.config.bias_drift_rate_rad_per_s > 0.0:
            self.bias_drift = SensorBiasDrift(
                drift_rate_rad_per_s=self.config.bias_drift_rate_rad_per_s,
                max_bias_rad=self.config.bias_drift_max_rad,
                seed=seed,
            )
        self._initialized = True

    def effective_bearing_std(
        self,
        base_std_rad: float,
        range_m: float,
    ) -> float:
        """Compute effective bearing noise std including range and atmosphere."""
        range_std = range_dependent_bearing_noise(base_std_rad, range_m, self.config)
        transmittance = atmospheric_attenuation(range_m, self.config)
        # Noise increases as transmittance drops
        atmo_multiplier = 1.0 / max(transmittance, 0.05)
        return min(range_std * atmo_multiplier, self.config.max_bearing_std_rad)

    def should_detect(
        self,
        rng: np.random.Generator,
        range_m: float,
    ) -> bool:
        """Stochastically decide if target is detected based on P_d."""
        transmittance = atmospheric_attenuation(range_m, self.config)
        pd = detection_probability(range_m, self.config, transmittance)
        return float(rng.random()) < pd

    def apply_bias(
        self,
        direction: np.ndarray,
        timestamp_s: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply bias drift to an observation direction vector."""
        if self.bias_drift is None:
            return direction
        bias = self.bias_drift.step(timestamp_s)
        return apply_bias_to_direction(direction, bias, rng)

    def generate_clutter(
        self,
        rng: np.random.Generator,
        node_position: np.ndarray,
        timestamp_s: float,
        node_id: str,
    ) -> List[BearingObservation]:
        """Generate false alarm observations for this sensor."""
        return generate_false_alarms(rng, node_position, self.config, timestamp_s, node_id)

    def reset(self) -> None:
        """Reset all stateful components."""
        if self.bias_drift is not None:
            self.bias_drift.reset()
        self._initialized = False


__all__ = [
    "SensorErrorConfig",
    "SensorModel",
    "SensorBiasDrift",
    "SENSOR_ERROR_PRESETS",
    "sensor_error_config_from_preset",
    "range_dependent_bearing_noise",
    "atmospheric_attenuation",
    "snr_at_range",
    "detection_probability",
    "generate_false_alarms",
    "apply_bias_to_direction",
]
