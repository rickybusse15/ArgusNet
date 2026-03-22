"""ArgusNet sensing sub-package.

Exports the public sensor API surface consumed by the rest of the platform.
"""

from argusnet.sensing.sensor_base import SensorBase
from argusnet.sensing.thermal import ThermalCameraModel

__all__ = [
    "SensorBase",
    "ThermalCameraModel",
]
