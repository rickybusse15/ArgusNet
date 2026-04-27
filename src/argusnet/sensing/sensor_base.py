"""SensorBase abstract base class for ArgusNet sensor models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["SensorBase"]


class SensorBase(ABC):
    """Abstract base class for all ArgusNet sensor models.

    Concrete sensor implementations must subclass this and implement all
    abstract methods.  The class is deliberately kept independent of the
    frozen-dataclass pattern so that subclasses can use
    ``@dataclass(frozen=True)`` freely — Python allows a frozen dataclass to
    inherit from a plain ABC.
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def sensor_id(self) -> str:
        """Unique string identifier for this sensor instance."""

    # ------------------------------------------------------------------
    # Abstract sensor interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_observation(
        self,
        platform_pos: np.ndarray,
        target_pos: np.ndarray,
        timestamp_s: float,
    ) -> dict | None:
        """Generate a noisy observation of *target_pos* from *platform_pos*.

        Returns a dict of measurement fields on detection, or ``None`` when the
        sensor fails to detect the target (outside FOV, below Pd threshold, …).

        Args:
            platform_pos: Sensor position in world frame (metres, shape (3,)).
            target_pos:   True target position in world frame (metres, shape (3,)).
            timestamp_s:  Simulation time of the observation (seconds).

        Returns:
            A measurement dict (keys vary by sensor type) or ``None``.
        """

    @abstractmethod
    def detection_probability(self, range_m: float) -> float:
        """Probability of detecting a target at *range_m* metres.

        Args:
            range_m: Slant range to target (metres, >= 0).

        Returns:
            Detection probability in [0, 1].
        """

    @abstractmethod
    def effective_noise_std(self, range_m: float) -> float:
        """1-sigma position noise at *range_m* metres (metres equivalent).

        Args:
            range_m: Slant range to target (metres, >= 0).

        Returns:
            Standard deviation of the measurement noise (metres).
        """

    @abstractmethod
    def in_fov(self, bearing_vec: np.ndarray) -> bool:
        """Return True if *bearing_vec* falls within the sensor field-of-view.

        Args:
            bearing_vec: Unit vector from sensor to target in sensor/world frame,
                         shape (3,).

        Returns:
            True if the bearing is within the FOV, False otherwise.
        """

    # ------------------------------------------------------------------
    # Concrete helpers (sensible no-op / safe defaults)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset any internal sensor state.

        The default implementation is a no-op.  Stateful sensors (e.g. those
        maintaining integration buffers or track histories) should override
        this to clear their state.
        """
        return None

    def is_degraded(self, weather_visibility: float) -> bool:  # noqa: ARG002
        """Return True if the sensor is operating in a degraded mode.

        Args:
            weather_visibility: Meteorological visibility in metres.  Lower
                values correspond to fog / rain / smoke.

        Returns:
            False by default.  Sensor subclasses may override this to return
            True when visibility drops below their operational threshold.
        """
        return False
