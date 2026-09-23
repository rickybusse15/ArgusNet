#!/usr/bin/env python3
"""Record the reference's motion-weather reads for every preset.

`WindModel.wind_at` and `WeatherModel.flight_speed_penalty` are the two
weather reads the motion path makes (`_apply_weather_adjustments`). The Rust
port (`argusnet-sim::weather`) receives each preset's *parameters* through the
scenario bundle, so the fixture records both: the parameters as the bundle
exporter writes them, and the reference's outputs over a grid of altitudes and
times that crosses zero altitude, a whole gust period, and each precipitation
drag clamp.

Usage:
    python3 scripts/capture_weather_fixture.py --out tests/fixtures/weather.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.simulation.scenario_bundle import motion_weather_document  # noqa: E402
from argusnet.world.weather import (  # noqa: E402
    WEATHER_PRESETS,
    PrecipitationModel,
    WeatherModel,
    WindModel,
)


def exact(value: float) -> str:
    return repr(float(value))


ALTITUDES = [-30.0, 0.0, 12.5, 80.0, 230.0, 612.0]
TIMES = [0.0, 1.0, 5.0, 7.25, 13.0, 20.0, 33.3]


def record(name: str, model: WeatherModel) -> dict:
    return {
        "name": name,
        "parameters": motion_weather_document(model),
        "winds": [
            {
                "altitude_m": exact(a),
                "time_s": exact(t),
                "wind": [exact(v) for v in model.wind.wind_at(a, t)],
            }
            for a in ALTITUDES
            for t in TIMES
        ],
        "penalties": [
            {"altitude_m": exact(a), "penalty": exact(model.flight_speed_penalty(a))}
            for a in ALTITUDES
        ],
    }


def capture() -> dict:
    cases = [record(name, model) for name, model in sorted(WEATHER_PRESETS.items())]
    # Each drag clamp, and the wind clamp at 0.1: a storm strong enough that
    # the wind term saturates, and precipitation heavy enough that each type's
    # floor binds.
    storm = WindModel(
        base_speed_mps=40.0,
        base_heading_rad=2.2,
        gust_amplitude_mps=8.0,
        gust_period_s=9.0,
        altitude_scaling=0.4,
    )
    for precip, rate in [("rain", 60.0), ("snow", 60.0), ("sleet", 60.0), ("fog", 0.0)]:
        cases.append(
            record(
                f"storm_{precip}",
                WeatherModel(
                    wind=storm, precipitation=PrecipitationModel(precip_type=precip, rate_mmph=rate)
                ),
            )
        )
    # Gusts disabled two ways: a zero period and a zero amplitude.
    cases.append(
        record(
            "no_gust_zero_period",
            WeatherModel(
                wind=WindModel(base_speed_mps=5.0, gust_amplitude_mps=3.0, gust_period_s=0.0)
            ),
        )
    )
    cases.append(
        record(
            "no_gust_zero_amplitude",
            WeatherModel(
                wind=WindModel(base_speed_mps=5.0, gust_amplitude_mps=0.0, gust_period_s=7.0)
            ),
        )
    )
    # A gust deeper than the base: speed would go negative and is floored at 0.
    cases.append(
        record(
            "gust_floors_speed_at_zero",
            WeatherModel(
                wind=WindModel(base_speed_mps=1.0, gust_amplitude_mps=4.0, gust_period_s=10.0)
            ),
        )
    )
    penalties = {c["name"]: [float(p["penalty"]) for p in c["penalties"]] for c in cases}
    assert min(penalties["storm_rain"]) < 0.5 * 0.9, "no storm case reached the drag floor"
    zero_speed = [
        w
        for w in next(c for c in cases if c["name"] == "gust_floors_speed_at_zero")["winds"]
        if w["wind"][0] in ("-0.0", "0.0") and w["wind"][1] in ("-0.0", "0.0")
    ]
    assert zero_speed, "the negative-speed floor never bound"
    return {"_source": "scripts/capture_weather_fixture.py", "cases": cases}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fixture = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(fixture['cases'])} weather cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
