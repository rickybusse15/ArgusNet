"""The P4.9c acceptance judge applies the envelopes PHASE_ACCEPTANCE.md declares.

`scripts/run_phase_acceptance.py` needs a release build and minutes of
simulation; its judgement does not. These runs are synthetic, so every
boundary the document declares can be put exactly where the test wants it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_phase_acceptance.py"
_spec = importlib.util.spec_from_file_location("run_phase_acceptance", SCRIPT)
acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acceptance)

METRICS = {
    "localisation_rmse_m": 40.0,
    "track_continuity_mean": 0.5,
    "time_to_reacquire_mean_s": 1.0,
    "covariance_reduction_mean": 0.8,
    "mission_completion_rate": 1.0,
    "energy_reserve_min": 0.8,
}


def _run(seed: int, *, rust=None, passed=(True, True), timing=None, reference_wall_s=1.0):
    identity = {"frames": 10, "node_ids": ["a"], "target_ids": ["t"]}
    reference = {**METRICS, "passed": passed[0], "required_objectives_total": 3}
    rust_report = {**METRICS, **(rust or {}), "passed": passed[1], "required_objectives_total": 3}
    return {
        "seed": seed,
        "reference": reference,
        "rust": rust_report,
        "reference_identity": identity,
        "rust_identity": dict(identity),
        "reference_wall_s": reference_wall_s,
        "rust_timing": timing or {"tick_ms": {"p95": 0.1}, "startup_ms": 20.0, "total_ms": 120.0},
    }


def _check(verdict: dict, name: str) -> dict:
    (found,) = [c for c in verdict["checks"] if c["check"] == name]
    return found


def test_identical_runs_pass():
    verdict = acceptance.judge("s", [_run(seed) for seed in range(5)])
    assert verdict["passed"], [c for c in verdict["checks"] if not c["passed"]]


@pytest.mark.parametrize(
    ("rmse", "passes"),
    # 25% of 40 m is 10 m, above the 2 m floor.
    [(50.0, True), (50.001, False), (30.0, True), (29.999, False)],
)
def test_rmse_envelope_is_a_quarter_of_the_reference(rmse, passes):
    runs = [_run(seed, rust={"localisation_rmse_m": rmse}) for seed in range(5)]
    assert _check(acceptance.judge("s", runs), "localisation_rmse_m")["passed"] is passes


def test_small_references_fall_back_to_the_absolute_floor():
    # 25% of 4 m is 1 m, so the 2 m floor governs.
    def run(seed, value):
        entry = _run(seed, rust={"localisation_rmse_m": value})
        entry["reference"]["localisation_rmse_m"] = 4.0
        return entry

    assert _check(acceptance.judge("s", [run(s, 6.0) for s in range(5)]), "localisation_rmse_m")[
        "passed"
    ]
    assert not _check(
        acceptance.judge("s", [run(s, 6.001) for s in range(5)]), "localisation_rmse_m"
    )["passed"]


def test_absolute_only_envelopes_ignore_the_reference_scale():
    runs = [_run(seed, rust={"track_continuity_mean": 0.601}) for seed in range(5)]
    assert not _check(acceptance.judge("s", runs), "track_continuity_mean")["passed"]


def test_a_metric_one_side_lacks_fails():
    runs = [_run(seed, rust={"localisation_rmse_m": None}) for seed in range(5)]
    assert not _check(acceptance.judge("s", runs), "localisation_rmse_m")["passed"]


def test_verdicts_must_agree_on_four_of_five_seeds():
    one_off = [_run(0, passed=(True, False))] + [_run(seed) for seed in range(1, 5)]
    two_off = [_run(seed, passed=(True, False)) for seed in range(2)] + [
        _run(seed) for seed in range(2, 5)
    ]
    assert _check(acceptance.judge("s", one_off), "check_pass_fail agreement")["passed"]
    assert not _check(acceptance.judge("s", two_off), "check_pass_fail agreement")["passed"]


def test_exact_checks_fail_on_any_identity_difference():
    runs = [_run(seed) for seed in range(5)]
    runs[3]["rust_identity"]["frames"] = 11
    verdict = acceptance.judge("s", runs)
    assert not _check(verdict, "seed 3 frames")["passed"]
    assert not verdict["passed"]


def test_speed_up_compares_the_tick_loop_not_the_process():
    # 1.0 s reference loop; Rust ran 220 ms in all, 20 ms of it startup, so
    # its loop took 200 ms: exactly 5x. Counting startup would make it 4.5x.
    timing = {"tick_ms": {"p95": 0.1}, "startup_ms": 20.0, "total_ms": 220.0}
    at_limit = [_run(seed, timing=timing) for seed in range(5)]
    assert _check(acceptance.judge("s", at_limit), "speed-up")["passed"]
    slower = {**timing, "total_ms": 220.1}
    assert not _check(
        acceptance.judge("s", [_run(s, timing=slower) for s in range(5)]), "speed-up"
    )["passed"]


def test_the_worst_seed_sets_the_tick_and_startup_gates():
    runs = [_run(seed) for seed in range(5)]
    runs[2]["rust_timing"] = {"tick_ms": {"p95": 5.001}, "startup_ms": 20.0, "total_ms": 120.0}
    runs[4]["rust_timing"] = {"tick_ms": {"p95": 0.1}, "startup_ms": 2000.1, "total_ms": 2100.1}
    verdict = acceptance.judge("s", runs)
    assert not _check(verdict, "tick p95")["passed"]
    assert not _check(verdict, "startup")["passed"]


def test_output_only_flags_are_dropped_from_the_rust_side(tmp_path):
    (config,) = [c for c in acceptance.CANONICAL_SCENARIOS if c.name == "persistent_long"]
    reference = acceptance._sim_argv(config, 7, tmp_path / "r.json")
    assert "--streaming" in reference and "--max-frames-in-memory" in reference
    argv = acceptance.rust_argv(config, 7, tmp_path / "r.json", tmp_path / "c.json")
    assert "--streaming" not in argv
    assert "--max-frames-in-memory" not in argv
    # The flag's value goes with it; everything else survives in order.
    kept = [t for t in reference if t not in ("--streaming", "--max-frames-in-memory", "500")]
    assert argv == [*kept, "--export-config", str(tmp_path / "c.json")]
