"""F4: scenario comparisons over paired seeds."""

from __future__ import annotations

import math

import pytest

from argusnet.evaluation.comparison import (
    METRICS,
    RunResult,
    compare,
    dispersion,
    markdown,
    paired,
    t975,
)


def run(variant, seed, rmse=None, error=None, passed=True, wall=1.0):
    return RunResult(
        variant=variant,
        seed=seed,
        metrics={"localisation_rmse_m": rmse, "mission_completion_rate": 1.0},
        error=error,
        passed_envelope=None if error else passed,
        wall_s=None if error else wall,
        tick_mean_ms=None if error else 0.2,
    )


def test_t_quantiles_are_tabulated_and_conservative():
    assert t975(1) == 12.706
    assert t975(29) == 2.045
    # Between tabulated values it rounds down the df, so the interval is wider.
    assert t975(35) == t975(30)
    assert t975(500) == 1.960
    with pytest.raises(ValueError):
        t975(0)


def test_dispersion_of_a_known_sample():
    d = dispersion([1.0, 2.0, 3.0, 4.0])
    assert d["n"] == 4
    assert d["mean"] == 2.5
    assert d["median"] == 2.5
    assert d["sd"] == pytest.approx(math.sqrt(5 / 3))
    assert (d["min"], d["max"]) == (1.0, 4.0)
    empty = dispersion([])
    assert empty["n"] == 0
    assert all(empty[k] is None for k in ("mean", "median", "sd", "min", "max"))


def test_a_consistent_improvement_is_detected_by_pairing():
    # Seed luck dominates the spread (10 to 40 m), but every seed improves by
    # 1 to 1.4 m: unpaired means could not see it, the pairs do.
    base = {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    var = {1: 9.0, 2: 18.8, 3: 28.6, 4: 39.0}
    p = paired(base, var, "lower")
    assert p["n_pairs"] == 4
    assert p["mean_diff"] == pytest.approx(-1.15)
    diffs = [-1.0, -1.2, -1.4, -1.0]
    sd = math.sqrt(sum((d + 1.15) ** 2 for d in diffs) / 3)
    assert p["sd_diff"] == pytest.approx(sd)
    half = 3.182 * sd / 2
    assert p["ci95"] == pytest.approx([-1.15 - half, -1.15 + half])
    assert (p["improved"], p["worsened"], p["tied"]) == (4, 0, 0)
    assert p["verdict"] == "better"


def test_direction_decides_better_or_worse():
    base = {1: 0.5, 2: 0.6, 3: 0.7}
    up = {1: 0.6, 2: 0.71, 3: 0.79}
    assert paired(base, up, "higher")["verdict"] == "better"
    assert paired(base, up, "lower")["verdict"] == "worse"
    # No preferred direction: reported, not scored.
    assert "verdict" not in paired(base, up, None)


def test_noise_is_no_detectable_difference_and_one_pair_is_too_few():
    base = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    var = {1: 1.5, 2: 0.4, 3: 1.6, 4: 0.6}
    assert paired(base, var, "lower")["verdict"] == "no detectable difference"
    assert paired({1: 1.0}, {1: 0.5}, "lower")["verdict"] == "too few pairs"


def test_only_seeds_both_variants_measured_are_paired():
    base = {1: 1.0, 2: None, 3: 3.0, 4: float("nan")}
    p = paired(base, {1: 2.0, 2: 2.0, 3: 4.0, 5: 9.0}, "lower")
    assert p["n_pairs"] == 2
    assert p["mean_diff"] == 1.0


def test_failed_runs_are_counted_not_dropped():
    runs = [
        run("base", 1, 10.0),
        run("base", 2, 12.0, passed=False),
        run("more", 1, 9.0),
        run("more", 2, error="argusnet-rs exited 1: tick 40 faulted"),
    ]
    report = compare(runs, ["base", "more"], metrics=["localisation_rmse_m"])
    assert report["sample_size"] == 2
    assert report["summary"]["base"]["outside_envelope"] == 1
    more = report["summary"]["more"]
    assert (more["runs"], more["failed"]) == (2, 1)
    assert more["failures"] == {2: "argusnet-rs exited 1: tick 40 faulted"}
    assert more["metrics"]["localisation_rmse_m"]["n"] == 1
    # The failed seed has no pair.
    assert report["paired_against_baseline"]["more"]["localisation_rmse_m"]["n_pairs"] == 1
    assert report["metric_definitions"]["localisation_rmse_m"]["better"] == "lower"


def test_a_run_for_an_unknown_variant_is_an_error():
    with pytest.raises(ValueError):
        compare([run("other", 1, 1.0)], ["base"])
    with pytest.raises(ValueError):
        compare([], [])


def test_every_metric_has_a_definition_and_a_direction_or_none():
    for metric, (better, definition) in METRICS.items():
        assert better in ("higher", "lower", None), metric
        assert len(definition) > 20, metric


def test_markdown_names_the_variants_metrics_and_failures():
    runs = [run("base", s, 10.0 + s) for s in (1, 2, 3)] + [
        run("more", 1, 9.5),
        run("more", 2, 10.6),
        run("more", 3, error="unsupported: scan mission with 0 drones"),
    ]
    report = compare(runs, ["base", "more"], metrics=["localisation_rmse_m"])
    report.update(
        scenario="baseline_coverage",
        seed_set="canonical",
        variant_args={"base": [], "more": ["--drone-count", "6"]},
        identity={
            "git_commit": "abc123",
            "git_dirty": False,
            "binary_sha256": "f" * 64,
            "world": [],
        },
    )
    text = markdown(report)
    assert "# Scenario comparison: baseline_coverage" in text
    assert "`--drone-count 6`" in text
    assert "(scenario defaults)" in text
    assert "`localisation_rmse_m` (lower is better)" in text
    assert "Paired against **base**" in text
    assert "more, seed 3: unsupported: scan mission with 0 drones" in text


def test_a_rounding_difference_is_a_tie_not_a_change():
    base = {s: 0.85 for s in range(1, 11)}
    var = {s: 0.85 - (3.7e-17 if s % 3 == 0 else 0.0) for s in range(1, 11)}
    p = paired(base, var, "higher")
    assert (p["improved"], p["worsened"], p["tied"]) == (0, 0, 10)
    assert p["verdict"] == "no detectable difference"
