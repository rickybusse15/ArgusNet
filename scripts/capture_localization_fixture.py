#!/usr/bin/env python3
"""Record GridLocalizer steps so the Rust port can be checked value-for-value.

P4.2: *"First validate all non-random formulas using recorded offsets."* This
produces those recordings.

Every step captures the estimator's full input — including the particle offsets
numpy drew and the footprint scores the native scorer returned — alongside the
estimate Python produced from them. With those recorded, the Rust estimator is a
pure function of the fixture, so the two can be compared directly rather than
statistically. The random draw is the only thing not being validated here, and
that is deliberate: migrating it is P4.2b's job, under ADR-006's envelopes.

The paths exercised are the ones with distinct behaviour, not just distinct
inputs: the no-map early return, the ordinary matching path, the confidence
blend across steps, the timeout, and multi-platform fusion.

Usage:
    python3 scripts/capture_localization_fixture.py --out tests/fixtures/localization_steps.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.localization import engine as engine_module  # noqa: E402
from argusnet.localization.engine import GridLocalizer, LocalizationConfig  # noqa: E402
from argusnet.mapping.coverage import CoverageMap  # noqa: E402
from argusnet.mapping.occupancy import GridBounds  # noqa: E402


class Recorder:
    """Captures the offsets drawn and the scores returned, per step."""

    def __init__(self) -> None:
        self.offsets: list[list[float]] | None = None
        self.scores: list[float] | None = None


def instrument(localizer: GridLocalizer, recorder: Recorder):
    """Wrap the two non-deterministic inputs so each step's values are kept."""
    real_rng = localizer._rng  # noqa: SLF001 - recording the reference is the point
    real_score = engine_module.score_footprints

    class RecordingRng:
        def normal(self, loc, scale, size):
            drawn = real_rng.normal(loc, scale, size)
            recorder.offsets = [[float(x), float(y)] for x, y in drawn]
            return drawn

    def recording_score(*args, **kwargs):
        result = real_score(*args, **kwargs)
        recorder.scores = [float(value) for value in np.asarray(result).ravel()]
        return result

    localizer._rng = RecordingRng()  # noqa: SLF001
    engine_module.score_footprints = recording_score
    return lambda: setattr(engine_module, "score_footprints", real_score)


def build_coverage(fraction_target: float) -> CoverageMap:
    """A coverage map filled to roughly `fraction_target`.

    Built by marking real circular footprints rather than by writing counts
    directly, so the scores the fixture records come from the same code path
    the simulation uses.
    """
    bounds = GridBounds(x_min_m=0.0, x_max_m=1000.0, y_min_m=0.0, y_max_m=1000.0, resolution_m=25.0)
    coverage = CoverageMap(bounds)
    if fraction_target <= 0.0:
        return coverage
    # Sweep a footprint across the map until coverage passes the target.
    radius = 90.0
    for y in np.arange(50.0, 1000.0, 90.0):
        for x in np.arange(50.0, 1000.0, 90.0):
            if coverage.stats.coverage_fraction >= fraction_target:
                return coverage
            coverage.mark_circular((float(x), float(y)), radius)
    return coverage


def capture() -> dict:
    cases: list[dict] = []

    def record_scenario(name: str, config: LocalizationConfig, plan: list[tuple[str, float]]):
        """Run `plan` as (platform_id, coverage_fraction) steps and record each."""
        localizer = GridLocalizer(config)
        recorder = Recorder()
        restore = instrument(localizer, recorder)
        steps: list[dict] = []
        try:
            for index, (platform_id, coverage_fraction) in enumerate(plan):
                coverage = build_coverage(coverage_fraction)
                actual_fraction = float(coverage.stats.coverage_fraction)
                nominal = [500.0 + 3.0 * index, 400.0 - 2.0 * index, 60.0]
                heading = 0.25 * index
                timestamp = 0.1 * index
                recorder.offsets = None
                recorder.scores = None

                estimate = localizer.update(
                    drone_id=platform_id,
                    nominal_position=np.array(nominal, dtype=float),
                    heading_rad=heading,
                    coverage_map=coverage,
                    footprint_radius_m=120.0,
                    timestamp_s=timestamp,
                )

                steps.append(
                    {
                        "platform_id": platform_id,
                        "nominal_position": nominal,
                        "heading_rad": heading,
                        "timestamp_s": timestamp,
                        "coverage_fraction": actual_fraction,
                        # None on the no-map path, where neither is reached.
                        "offsets": recorder.offsets,
                        "scores": recorder.scores,
                        "expected": {
                            "position_estimate": [float(v) for v in estimate.position_estimate],
                            "position_std_m": float(estimate.position_std_m),
                            "confidence": float(estimate.confidence),
                        },
                    }
                )
        finally:
            restore()

        fused = localizer.fuse_estimates([platform_id for platform_id, _ in plan])
        cases.append(
            {
                "name": name,
                "config": {
                    "search_radius_m": config.search_radius_m,
                    "particle_count": config.particle_count,
                    "convergence_threshold_m": config.convergence_threshold_m,
                    "min_coverage_to_localize": config.min_coverage_to_localize,
                    "confidence_decay": config.confidence_decay,
                    "annealing_rate": config.annealing_rate,
                    "localization_timeout_steps": config.localization_timeout_steps,
                },
                "steps": steps,
                "expected_fused": None
                if fused is None
                else {
                    "position_estimate": [float(v) for v in fused.position_estimate],
                    "position_std_m": float(fused.position_std_m),
                    "confidence": float(fused.confidence),
                },
                "expected_any_timed_out": bool(localizer.any_timed_out),
            }
        )

    default = LocalizationConfig()

    # The no-map early return, then the same platform once coverage arrives, so
    # the stored zero-confidence estimate feeds the next step's blend.
    record_scenario(
        "no_map_then_matching",
        default,
        [("drone-1", 0.0), ("drone-1", 0.0), ("drone-1", 0.6), ("drone-1", 0.6)],
    )

    # A single platform over enough steps for annealing to bite.
    record_scenario("annealing", default, [("drone-1", 0.5)] * 8)

    # Two platforms interleaved, so per-platform step counts must stay separate,
    # and both end converged enough to fuse.
    record_scenario(
        "two_platforms_fused",
        default,
        [("drone-1", 0.7), ("drone-2", 0.7), ("drone-1", 0.7), ("drone-2", 0.7)],
    )

    # The timeout path, with a budget short enough to reach.
    record_scenario(
        "timeout",
        LocalizationConfig(localization_timeout_steps=3),
        [("drone-1", 0.5)] * 4,
    )

    return {
        "source": "src/argusnet/localization/engine.py::GridLocalizer",
        "note": (
            "Recorded particle offsets, footprint scores and the estimates Python "
            "produced from them. With the random draw recorded rather than repeated, "
            "the Rust estimator is a pure function of this fixture and can be compared "
            "value-for-value. Migrating the draw itself is P4.2b (ADR-006)."
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    document = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    total = sum(len(case["steps"]) for case in document["cases"])
    print(f"wrote {args.out}: {len(document['cases'])} cases, {total} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
