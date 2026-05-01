from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from argusnet.core.errors import ValidationError


def assert_finite(arr: object, *, name: str = "array") -> None:
    values = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValidationError(f"{name} must contain only finite values.")


def assert_unit_interval(p: float, *, name: str = "value") -> None:
    value = float(p)
    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValidationError(f"{name} must be finite and within [0, 1].")


def assert_monotonic_timestamps(stream: Iterable[float], *, name: str = "timestamps") -> None:
    previous: float | None = None
    for index, raw in enumerate(stream):
        timestamp = float(raw)
        if not np.isfinite(timestamp):
            raise ValidationError(f"{name}[{index}] must be finite.")
        if previous is not None and timestamp < previous:
            raise ValidationError(f"{name} must be monotonically non-decreasing.")
        previous = timestamp

