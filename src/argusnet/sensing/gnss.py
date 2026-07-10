"""GNSS position noise model for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3, vec3

__all__ = [
    "GNSSModel",
    "GNSSMeasurement",
    "GNSSSimulator",
    "OutageSchedule",
    "sample_gnss_position",
]


@dataclass(frozen=True)
class GNSSModel:
    """GNSS receiver noise model based on DOP (dilution of precision).

    Horizontal and vertical 1-sigma errors are derived from a base CEP
    (circular error probable) value multiplied by HDOP and VDOP.
    """

    base_cep_m: float = 2.5
    """Base circular error probable at HDOP=1 (metres, 50th percentile)."""

    hdop: float = 1.0
    """Horizontal dilution of precision."""

    vdop: float = 1.5
    """Vertical dilution of precision."""

    # CEP → 1-sigma conversion factor for 2-D Gaussian: CEP ≈ 1.177 * σ
    _CEP_TO_SIGMA: float = 1.0 / 1.177

    @property
    def horizontal_sigma_m(self) -> float:
        """1-sigma horizontal position error (metres)."""
        return self.base_cep_m * self.hdop * self._CEP_TO_SIGMA

    @property
    def vertical_sigma_m(self) -> float:
        """1-sigma vertical position error (metres)."""
        return self.base_cep_m * self.vdop * self._CEP_TO_SIGMA

    def sample_error(self, rng: np.random.Generator | None = None) -> Vector3:
        """Sample a 3-D position error vector (dx, dy, dz) in metres."""
        if rng is None:
            rng = np.random.default_rng()
        h_sigma = self.horizontal_sigma_m
        v_sigma = self.vertical_sigma_m
        dx = float(rng.normal(0.0, h_sigma))
        dy = float(rng.normal(0.0, h_sigma))
        dz = float(rng.normal(0.0, v_sigma))
        return vec3(dx, dy, dz)


@dataclass(frozen=True)
class GNSSMeasurement:
    """A noisy GNSS position fix."""

    position: Vector3
    """Reported position in ENU metres."""

    timestamp_s: float
    """Time of fix (simulation seconds)."""

    h_sigma_m: float
    """1-sigma horizontal accuracy estimate (metres)."""

    v_sigma_m: float
    """1-sigma vertical accuracy estimate (metres)."""


def sample_gnss_position(
    true_position: Vector3,
    model: GNSSModel,
    timestamp_s: float,
    rng: np.random.Generator | None = None,
) -> GNSSMeasurement:
    """Generate a noisy GNSS fix from a ground-truth position."""
    error = model.sample_error(rng)
    noisy_pos = np.asarray(true_position, dtype=float) + error
    return GNSSMeasurement(
        position=noisy_pos,
        timestamp_s=timestamp_s,
        h_sigma_m=model.horizontal_sigma_m,
        v_sigma_m=model.vertical_sigma_m,
    )


@dataclass(frozen=True)
class OutageSchedule:
    """Deterministic GNSS outage windows generated from a seed.

    Outage start times follow a Poisson process at *rate_per_hour* over
    [0, horizon_s]; each outage lasts a uniform duration between
    *min_duration_s* and *max_duration_s*. The schedule is fully determined
    by the seed, so repeated runs see identical outages.
    """

    rate_per_hour: float = 2.0
    min_duration_s: float = 5.0
    max_duration_s: float = 30.0
    horizon_s: float = 3600.0
    seed: int = 0

    def windows(self) -> tuple[tuple[float, float], ...]:
        rng = np.random.default_rng(self.seed)
        expected = self.rate_per_hour * self.horizon_s / 3600.0
        count = int(rng.poisson(max(expected, 0.0)))
        starts = np.sort(rng.uniform(0.0, self.horizon_s, size=count))
        durations = rng.uniform(self.min_duration_s, self.max_duration_s, size=count)
        return tuple(
            (float(start), float(start + duration))
            for start, duration in zip(starts, durations, strict=True)
        )

    def is_out(self, timestamp_s: float) -> bool:
        return any(start <= timestamp_s < end for start, end in self.windows())


class GNSSSimulator:
    """Stateful GNSS fix generator with colored noise and optional outages.

    Layers on top of the white-noise :class:`GNSSModel`:

    - A first-order Gauss-Markov (exponentially correlated) bias per axis,
      capturing slowly varying multipath/atmospheric error. Disabled when
      ``gm_sigma_m`` is 0.
    - Optional deterministic outage windows during which ``sample`` returns
      ``None`` (no fix).

    All randomness is drawn from a single seeded generator, so a fixed seed
    reproduces the same fix sequence. Timestamps must be non-decreasing.
    """

    def __init__(
        self,
        model: GNSSModel | None = None,
        *,
        gm_sigma_m: float = 0.0,
        gm_time_constant_s: float = 60.0,
        outages: OutageSchedule | None = None,
        seed: int = 0,
    ) -> None:
        self.model = model if model is not None else GNSSModel()
        self.gm_sigma_m = float(gm_sigma_m)
        self.gm_time_constant_s = max(float(gm_time_constant_s), 1e-3)
        self.outages = outages
        self._seed = int(seed)
        self._rng = np.random.default_rng(seed)
        self._outage_windows = outages.windows() if outages is not None else ()
        self._gm_bias: np.ndarray = np.zeros(3, dtype=float)
        self._last_t: float | None = None

    def in_outage(self, timestamp_s: float) -> bool:
        return any(start <= timestamp_s < end for start, end in self._outage_windows)

    def sample(self, true_position: Vector3, timestamp_s: float) -> GNSSMeasurement | None:
        """Return a fix for *timestamp_s*, or ``None`` during an outage.

        The Gauss-Markov bias is propagated across outages so error remains
        correlated when fixes resume.
        """
        if self.gm_sigma_m > 0.0:
            dt = 0.0 if self._last_t is None else max(float(timestamp_s) - self._last_t, 0.0)
            decay = float(np.exp(-dt / self.gm_time_constant_s))
            drive = self.gm_sigma_m * float(np.sqrt(max(1.0 - decay * decay, 0.0)))
            self._gm_bias = decay * self._gm_bias + drive * self._rng.standard_normal(3)
        self._last_t = float(timestamp_s)

        if self.in_outage(timestamp_s):
            return None

        white = np.asarray(self.model.sample_error(self._rng), dtype=float)
        noisy_pos = np.asarray(true_position, dtype=float) + white + self._gm_bias
        return GNSSMeasurement(
            position=noisy_pos,
            timestamp_s=float(timestamp_s),
            h_sigma_m=self.model.horizontal_sigma_m,
            v_sigma_m=self.model.vertical_sigma_m,
        )

    def reset_state(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self._gm_bias = np.zeros(3, dtype=float)
        self._last_t = None
