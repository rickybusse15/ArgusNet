#!/usr/bin/env python3
"""Record the Python sensor error model so the Rust port can be checked.

Same shape as `capture_localization_fixture.py`, and for the same reason: the
formulas are validated against recorded values before anything stochastic moves.

Two kinds of case:

- **closed form** — `range_dependent_bearing_noise`, `atmospheric_attenuation`,
  `snr_at_range` and `detection_probability` evaluated across ranges that
  straddle every branch: below and above the detection knee, at and past the
  noise cap, at the reference range, and at the degenerate inputs each function
  special-cases.
- **drift** — `SensorBiasDrift.step` driven over a sequence of timestamps with
  the increment numpy drew recorded alongside the resulting bias, so the Rust
  state machine can be replayed against it exactly.

Usage:
    python3 scripts/capture_sensor_noise_fixture.py --out tests/fixtures/sensor_noise.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.sensing.models.noise import (  # noqa: E402
    SensorBiasDrift,
    SensorErrorConfig,
    atmospheric_attenuation,
    detection_probability,
    range_dependent_bearing_noise,
    snr_at_range,
)

CONFIG_FIELDS = (
    "base_bearing_std_rad",
    "range_noise_exponent",
    "range_noise_reference_m",
    "max_bearing_std_rad",
    "atmospheric_attenuation_coeff",
    "min_detection_probability",
    "detection_range_knee_m",
    "detection_range_falloff_m",
    "bias_drift_rate_rad_per_s",
    "bias_drift_max_rad",
    "snr_reference_db",
    "snr_range_exponent",
)


def exact(value: float) -> str:
    """Render a float as a string so it survives the trip into Rust.

    `serde_json`'s number parser rounds the last bit: it reads
    `0.9999000049998333` back as `...332`, while Rust's own `str::parse::<f64>`
    reads it exactly. Since the whole point of these fixtures is value-for-value
    comparison, the numbers that matter are written as strings and parsed on the
    Rust side with `str::parse`. `repr` is shortest-round-trip, so no precision
    is lost in the rendering itself.
    """
    return repr(float(value))


def config_dict(config: SensorErrorConfig) -> dict:
    return {name: exact(getattr(config, name)) for name in CONFIG_FIELDS}


# Ranges chosen to straddle every branch rather than to be evenly spaced:
# the degenerate cases each function special-cases, both sides of the detection
# knee, the noise reference range, and far enough out to hit both clamps.
RANGES = [
    -100.0,
    0.0,
    1.0,
    10.0,
    250.0,
    799.9,
    800.0,
    800.1,
    1000.0,
    1200.0,
    2000.0,
    5000.0,
    50_000.0,
    1_000_000.0,
]

TRANSMITTANCES = [0.0, 0.25, 1.0]


def closed_form_cases() -> list[dict]:
    configs = {
        "default": SensorErrorConfig(),
        # Attenuation off, so detection is not scaled by transmittance.
        "no_attenuation": SensorErrorConfig(atmospheric_attenuation_coeff=0.0),
        # A zero falloff collapses detection straight to the floor past the knee.
        "sharp_cutoff": SensorErrorConfig(detection_range_falloff_m=0.0),
        # A degenerate reference range takes the early return in three functions.
        "zero_reference": SensorErrorConfig(range_noise_reference_m=0.0),
        "steep_noise": SensorErrorConfig(range_noise_exponent=3.0, max_bearing_std_rad=1.0),
    }

    cases = []
    for name, config in configs.items():
        rows = []
        for range_m in RANGES:
            transmittance = atmospheric_attenuation(range_m, config)
            row = {
                "range_m": exact(range_m),
                "bearing_std_rad": exact(
                    range_dependent_bearing_noise(config.base_bearing_std_rad, range_m, config)
                ),
                "transmittance": exact(transmittance),
                "snr_db": exact(snr_at_range(range_m, config)),
                "detection_probability": {
                    repr(t): exact(detection_probability(range_m, config, t))
                    for t in TRANSMITTANCES
                },
            }
            rows.append(row)
        cases.append({"name": name, "config": config_dict(config), "rows": rows})
    return cases


def drift_cases() -> list[dict]:
    """Replay `SensorBiasDrift.step`, recording the increment numpy drew."""
    scenarios = [
        # Ordinary walk over even ticks.
        ("even_ticks", 1e-4, 0.01, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]),
        # Uneven intervals, so the sqrt(dt) scaling is exercised.
        ("uneven_ticks", 1e-4, 0.01, [0.0, 0.05, 0.5, 5.0, 5.0, 50.0]),
        # A tight clamp the walk should reach and hold against.
        ("tight_clamp", 5e-3, 1e-4, [float(i) for i in range(12)]),
        # Time repeating and running backwards -- both clamp dt to zero.
        ("non_monotonic", 1e-4, 0.01, [0.0, 1.0, 1.0, 0.5, 2.0]),
        # Drift disabled: every step returns zero and records no baseline.
        ("disabled", 0.0, 0.01, [0.0, 1.0, 2.0]),
    ]

    cases = []
    for name, rate, max_bias, timestamps in scenarios:
        drift = SensorBiasDrift(drift_rate_rad_per_s=rate, max_bias_rad=max_bias, seed=7)
        real_rng = drift._rng  # noqa: SLF001 - recording the reference is the point
        drawn: list[float] = []

        class RecordingRng:
            """numpy's Generator methods are read-only, so wrap the object.

            Default arguments bind the loop's values at class-definition time;
            without them every scenario's recorder would close over the last
            iteration's generator (ruff B023).
            """

            def normal(self, loc, scale, _rng=real_rng, _drawn=drawn):
                value = _rng.normal(loc, scale)
                _drawn.append(float(value))
                return value

        drift._rng = RecordingRng()  # noqa: SLF001

        steps = []
        for timestamp in timestamps:
            before = len(drawn)
            bias = drift.step(timestamp)
            # None when the step took a path that drew nothing: no baseline yet,
            # zero elapsed time, or drift disabled.
            increment = drawn[before] if len(drawn) > before else None
            steps.append(
                {
                    "timestamp_s": exact(timestamp),
                    "increment": None if increment is None else exact(increment),
                    "expected_bias_rad": exact(bias),
                }
            )

        cases.append(
            {
                "name": name,
                "drift_rate_rad_per_s": exact(rate),
                "max_bias_rad": exact(max_bias),
                "steps": steps,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    document = {
        "source": "src/argusnet/sensing/models/noise.py",
        "note": (
            "Recorded values from the Python sensor error model. The closed-form "
            "cases are pure functions of their inputs. The drift cases record the "
            "increment numpy drew at each step, so the Rust state machine replays "
            "against the same numbers rather than being compared statistically."
        ),
        "closed_form": closed_form_cases(),
        "drift": drift_cases(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    rows = sum(len(case["rows"]) for case in document["closed_form"])
    steps = sum(len(case["steps"]) for case in document["drift"])
    print(f"wrote {args.out}: {rows} closed-form rows, {steps} drift steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
