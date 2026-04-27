"""Sensor latency and timestamp-jitter model for ArgusNet."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "LatencyModel",
    "apply_latency",
]


@dataclass(frozen=True)
class LatencyModel:
    """Models end-to-end sensor observation latency.

    Latency has two components:
    - A fixed propagation delay (*mean_s*)
    - Zero-mean Gaussian jitter (*jitter_std_s*)

    Timestamps are shifted backward by the sampled latency so that the
    reported observation time reflects when the event *occurred*, not when
    data arrived.
    """

    mean_s: float = 0.0
    """Mean latency (propagation delay) in seconds."""

    jitter_std_s: float = 0.0
    """Standard deviation of timestamp jitter (seconds)."""

    min_s: float = 0.0
    """Hard lower bound on sampled latency (must be >= 0)."""

    max_s: float = math.inf
    """Hard upper bound on sampled latency."""

    def sample(self, rng: np.random.Generator | None = None) -> float:
        """Sample a latency value from the model."""
        if rng is None:
            rng = np.random.default_rng()
        raw = self.mean_s
        if self.jitter_std_s > 0:
            raw += float(rng.normal(0.0, self.jitter_std_s))
        return float(np.clip(raw, self.min_s, self.max_s))


def apply_latency(
    timestamp_s: float,
    model: LatencyModel,
    rng: np.random.Generator | None = None,
) -> float:
    """Return a timestamp shifted backward by a latency sample."""
    return timestamp_s - model.sample(rng)
