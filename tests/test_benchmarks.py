from __future__ import annotations

import json

import pytest

from argusnet.evaluation.benchmarks import (
    DEFAULT_SCENARIO_SEEDS,
    build_performance_summary,
    compute_timing_summary,
    default_benchmark_configs,
    get_scenario_families,
    get_scenario_metadata,
    list_scenario_family_ids,
    normalize_command,
    write_performance_summary,
)
from argusnet.evaluation.metrics import EvaluationReport

_ENVIRONMENT = {
    "commit_sha": "abc123",
    "git_dirty": False,
    "os": {
        "platform": "test-os",
        "system": "Darwin",
        "release": "test-release",
        "version": "test-version",
        "machine": "arm64",
        "processor": "arm",
    },
    "python": {
        "version": "3.12.0",
        "implementation": "CPython",
        "executable": "/usr/bin/python",
    },
    "rust": {
        "rustc": "rustc 1.80.0",
        "cargo": "cargo 1.80.0",
    },
    "hardware": {
        "cpu_count": 8,
        "memory_total_bytes": 17_179_869_184,
    },
}


def test_scenario_families_match_documented_order_and_metadata() -> None:
    assert list_scenario_family_ids() == [
        "mapping_coverage",
        "localization_recovery",
        "inspection_workflow",
        "coordination_stress",
        "revisit_and_change",
    ]

    families = get_scenario_families()
    mapping = families["mapping_coverage"]
    assert [scenario.name for scenario in mapping.scenarios] == [
        "mapping_small",
        "mapping_medium_multi",
        "mapping_alpine",
    ]
    assert mapping.scenarios[2].map_preset == "medium"
    assert mapping.scenarios[2].terrain_preset == "alpine"
    assert mapping.scenarios[2].drone_count == 3
    assert mapping.scenarios[2].difficulty == pytest.approx(0.5)
    assert mapping.scenarios[2].mission_type == "mapping"
    assert all(scenario.seeds == DEFAULT_SCENARIO_SEEDS for scenario in mapping.scenarios)

    coordination = get_scenario_metadata("coordination_large_area")
    assert coordination.family == "coordination_stress"
    assert coordination.map_preset == "regional"
    assert coordination.mission_type == "map_localize_inspect"


def test_default_benchmark_configs_filter_and_override_runs() -> None:
    configs = default_benchmark_configs(
        ["coordination_stress"],
        seeds=[101],
        duration_s=15.5,
    )

    assert [config.name for config in configs] == [
        "coordination_two_drone",
        "coordination_dense_pois",
        "coordination_large_area",
    ]
    assert all(config.family == "coordination_stress" for config in configs)
    assert all(config.seeds == [101] for config in configs)
    assert all(config.duration_s == pytest.approx(15.5) for config in configs)
    assert configs[1].terrain_preset == "urban_flat"
    assert configs[1].drone_count == 4
    assert "coordination_stress" in configs[1].tags
    assert "no unresolved separation violations" in configs[1].pass_criteria


def test_timing_summary_is_deterministic_and_rejects_bad_values() -> None:
    summary = compute_timing_summary([0.4, 0.1, 0.2, 0.3])

    assert summary["count"] == 4
    assert summary["min_s"] == pytest.approx(0.1)
    assert summary["max_s"] == pytest.approx(0.4)
    assert summary["mean_s"] == pytest.approx(0.25)
    assert summary["p50_s"] == pytest.approx(0.25)
    assert summary["p95_s"] == pytest.approx(0.385)
    assert summary["p99_s"] == pytest.approx(0.397)
    assert summary["total_s"] == pytest.approx(1.0)

    empty = compute_timing_summary([])
    assert empty["count"] == 0
    assert empty["p95_s"] is None
    assert empty["p99_s"] is None

    with pytest.raises(ValueError, match="non-negative"):
        compute_timing_summary([0.1, -0.2])
    with pytest.raises(ValueError, match="finite"):
        compute_timing_summary([float("nan")])


def test_performance_summary_contains_command_timing_and_environment_metadata() -> None:
    report = EvaluationReport(
        scenario_name="mapping_small",
        mission_type="mapping",
        difficulty=0.1,
        seed=7,
        duration_s=12.0,
        passed=True,
        generated_at_utc="2026-04-29T00:00:00+00:00",
    )

    summary = build_performance_summary(
        scenario_name="mapping_small",
        family="mapping_coverage",
        seed=7,
        command=["argusnet", "benchmark", "--family", "mapping_coverage"],
        timings_s=[0.01, 0.02, 0.04],
        report=report,
        environment_metadata=_ENVIRONMENT,
        generated_at_utc="2026-04-29T12:00:00+00:00",
    )

    assert summary["schema_version"] == 1
    assert summary["scenario_name"] == "mapping_small"
    assert summary["family"] == "mapping_coverage"
    assert summary["seed"] == 7
    assert summary["command"] == ["argusnet", "benchmark", "--family", "mapping_coverage"]
    assert summary["commit_sha"] == "abc123"
    assert summary["generated_at_utc"] == "2026-04-29T12:00:00+00:00"
    assert summary["timing_s"]["p95_s"] == pytest.approx(0.038)
    assert summary["timing_s"]["p99_s"] == pytest.approx(0.0396)
    assert summary["environment"]["os"]["platform"] == "test-os"
    assert summary["environment"]["python"]["version"] == "3.12.0"
    assert summary["environment"]["rust"]["rustc"] == "rustc 1.80.0"
    assert summary["report"]["scenario_name"] == "mapping_small"
    assert summary["report"]["passed"] is True


def test_write_performance_summary_uses_standard_filename_for_directory(tmp_path) -> None:
    summary = build_performance_summary(
        scenario_name="revisit_known_poi",
        family="revisit_and_change",
        seed=29,
        command="argusnet benchmark --scenario revisit_known_poi",
        timings_s=[1.0],
        environment_metadata=_ENVIRONMENT,
        generated_at_utc="2026-04-29T12:30:00+00:00",
    )

    output_path = write_performance_summary(tmp_path, summary)

    assert output_path == tmp_path / "performance_summary.json"
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == summary
    assert loaded["command"] == ["argusnet benchmark --scenario revisit_known_poi"]


def test_normalize_command_handles_explicit_strings_and_sequences() -> None:
    assert normalize_command("argusnet benchmark") == ["argusnet benchmark"]
    assert normalize_command(("argusnet", "benchmark", 7)) == ["argusnet", "benchmark", "7"]
