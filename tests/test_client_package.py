from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from argusnet.cli.main import main
from argusnet.evaluation.assurance import build_client_package
from argusnet.evaluation.benchmarks import BenchmarkConfig, BenchmarkRun, aggregate_reports
from argusnet.evaluation.metrics import (
    check_pass_fail,
    evaluate_replay,
    report_from_dict,
    report_to_dict,
)
from argusnet.evaluation.reports import print_report, report_to_markdown
from argusnet.evaluation.scorecard import scorecard_from_replay


def replay(phase="complete", completed=1, battery=0.4, abort=False):
    doc = {
        "meta": {"scenario_name": "assurance", "seed": 7, "dt_s": 1.0},
        "frames": [
            {
                "timestamp_s": 0.0,
                "nodes": [{"node_id": "drone", "is_mobile": True, "battery_fraction": battery}],
                "scan_mission_state": {
                    "phase": phase,
                    "total_poi_count": 1,
                    "completed_poi_count": completed,
                    "localization_timed_out": True,
                    "poi_statuses": [
                        {"poi_id": "asset", "status": "complete" if completed else "pending"}
                    ],
                },
                "safety_events": [
                    {
                        "event_id": "safe-1",
                        "subject_id": "drone",
                        "timestamp_s": 0.0,
                        "reason": "clearance",
                    }
                ],
            }
        ],
    }
    if abort:
        # The safety monitor's abort, as the runtime records it.
        doc["frames"][0]["safety_events"].append(
            {
                "validation_id": "val-9",
                "subject_id": "drone",
                "accepted": False,
                "reason": "abort",
                "violations": ["min_terrain_clearance"],
            }
        )
    return doc


def test_missing_data_is_never_reported_as_success(capsys):
    doc = {"meta": {}, "frames": []}
    report = evaluate_replay(doc, "missing", "unknown", 0.0, 0)
    assert report.mission_completion_rate is None
    assert report.energy_reserve_min is None
    assert report.track_continuity_mean is None
    checked = check_pass_fail(report)
    assert not checked.passed
    # Unavailable continuity, accuracy and energy fail; no objectives is
    # not applicable to the completion check (never 100%, never a failure).
    assert len(checked.failure_reasons) == 3
    assert not any("mission_completion_rate" in r for r in checked.failure_reasons)
    assert scorecard_from_replay(doc).mission["completion_rate"] is None
    assert report_from_dict(report_to_dict(checked)) == checked
    assert "N/A" in report_to_markdown(report)
    print_report(report)
    assert "N/A" in capsys.readouterr().out


def test_aggregates_skip_unavailable_metrics():
    report = evaluate_replay({"frames": []}, "missing", "unknown", 0, 0)
    result = aggregate_reports("missing", [BenchmarkRun(BenchmarkConfig("missing"), 0, report)])
    assert result.mission_completion_mean is None
    assert result.track_continuity_mean is None


@pytest.mark.parametrize("battery", [None, -1, -0.5, 1.1, float("nan")])
def test_invalid_battery_does_not_become_full_reserve(battery):
    doc = replay(battery=battery)
    report = evaluate_replay(doc, "battery", "mapping", 0, 0)
    assert report.energy_reserve_min is None


def test_energy_reads_recorded_minimum_and_keeps_a_flat_battery():
    doc = replay(battery=0.9)
    doc["frames"].append(
        {
            "timestamp_s": 1,
            "nodes": [
                {"node_id": "drone", "is_mobile": True, "battery_fraction": 0.0},
                {"node_id": "tower", "is_mobile": False, "battery_fraction": -1},
            ],
        }
    )
    report = evaluate_replay(doc, "battery", "mapping", 0, 0)
    assert report.energy_reserve_min == 0
    assert report.energy_reserve_per_drone == {"drone": 0.0}


@pytest.mark.parametrize(
    "mission",
    [
        {"required_objectives_total": 3},
        {"required_objectives_met": 0},
        {"required_objectives_total": 0, "required_objectives_met": 0},
    ],
)
def test_unknown_or_empty_objectives_are_unavailable(mission):
    report = evaluate_replay({"mission": mission, "frames": []}, "missing", "unknown", 0, 0)
    assert report.mission_completion_rate is None


@pytest.mark.parametrize(
    "mission",
    [
        {"required_objectives_total": -1, "required_objectives_met": 0},
        {"required_objectives_total": 1, "required_objectives_met": 2},
        {"required_objectives_total": 1, "required_objectives_met": -1},
        {"required_objectives_total": 1, "required_objectives_met": True},
    ],
)
def test_invalid_objective_counts_are_refused(mission):
    with pytest.raises(ValueError, match="objective counts"):
        evaluate_replay({"mission": mission}, "invalid", "unknown", 0, 0)


@pytest.mark.parametrize(
    "phase,completed,abort,status",
    [
        ("complete", 1, False, "completed"),
        ("egress", 0, True, "aborted"),
        ("egress", 1, False, "incomplete"),
    ],
)
def test_package_records_outcomes_without_certifying_landing(phase, completed, abort, status):
    doc = replay(phase, completed, abort=abort)
    package = build_client_package(doc)
    report = json.loads(package.assurance)
    assert report["status"] == status
    assert report["completion_rate"] == completed
    assert report["landing_outcome"] is None
    assert report["localization_integrity"] is None
    assert report["objectives"][0]["evidence_sufficiency"] is None
    for event in report["events"]:
        _, _, frame, collection, index = event["source_pointer"].split("/")
        recorded = doc["frames"][int(frame)][collection][int(index)]
        assert recorded == event["record"]
    assert report["localization_timeout_sources"] == [
        "/frames/0/scan_mission_state/localization_timed_out",
    ]
    assert "qualification" in report["unavailable_fields"]


def test_package_bytes_hashes_and_regeneration_are_deterministic(tmp_path):
    doc = replay()
    original = copy.deepcopy(doc)
    package = build_client_package(doc)
    assert doc == original
    assert package == build_client_package(doc)
    regenerated = build_client_package(json.loads(package.replay))
    assert regenerated == package
    manifest = json.loads(package.manifest)
    package.write(tmp_path / "package")
    assert manifest["signature"] is None
    assert manifest["status"] == "incomplete"
    for artifact in manifest["artifacts"]:
        data = (tmp_path / "package" / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]
        assert len(data) == artifact["size_bytes"]
    with pytest.raises(FileExistsError):
        package.write(tmp_path / "package")


def test_client_package_cli_preserves_replay_schema(tmp_path, capsys):
    fixture = Path(__file__).parent / "fixtures" / "replay_contract_fixture.json"
    main(["client-package", "--replay", str(fixture), "--output", str(tmp_path / "package")])
    from argusnet.evaluation.replay import validate_replay_document

    generated = json.loads((tmp_path / "package" / "replay.json").read_text())
    validate_replay_document(generated)
    assert "mission_assurance" in generated["summary"]
    assert "Client package written" in capsys.readouterr().out


def test_one_platform_with_missing_energy_cannot_certify_the_fleet():
    doc = replay()
    doc["frames"][0]["nodes"].append({"node_id": "missing", "is_mobile": True})
    report = evaluate_replay(doc, "battery", "mapping", 0, 0)
    assert report.energy_reserve_per_drone == {"drone": 0.4}
    assert report.energy_reserve_min is None


def test_client_package_cli_refuses_invalid_replay_without_partial_output(tmp_path):
    source = tmp_path / "invalid.json"
    source.write_text('{"frames": []}')
    with pytest.raises(SystemExit, match="validation failed"):
        main(["client-package", "--replay", str(source), "--output", str(tmp_path / "output")])
    assert not (tmp_path / "output").exists()


def test_shared_python_rust_assurance_fixture_cannot_drift():
    fixture = Path(__file__).parent / "fixtures" / "mission_assurance_fixture.json"
    assert fixture.read_bytes() == build_client_package(replay()).assurance


@pytest.mark.parametrize(
    ("phase", "abort", "status", "outcomes"),
    [
        ("complete", False, "completed", {"completed"}),
        ("inspecting", True, "aborted", {"aborted"}),
        ("inspecting", False, "incomplete", {"completed", "objectives_unmet"}),
    ],
)
def test_the_report_carries_an_f1_outcome_that_agrees_with_its_status(
    phase, abort, status, outcomes
):
    """One outcome format: the report embeds the F1 record (reasons with the
    frames and events they rest on) instead of recomputing an outcome, and
    takes an abort from it."""
    doc = replay(phase, abort=abort)
    report = json.loads(build_client_package(doc).assurance)
    assert report["status"] == status
    assert report["outcome"]["format"] == "argusnet.mission-outcome/1"
    assert report["outcome"]["outcome"] in outcomes
    assert all(reason["code"] and reason["detail"] for reason in report["outcome"]["reasons"])


def test_the_source_hash_is_the_input_as_given():
    """The hash is over the replay as given (canonical JSON), with no
    summary added first; a package of a package's replay names the same
    source."""
    doc = replay()
    assert "summary" not in doc
    canonical = (json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    first = json.loads(build_client_package(doc).assurance)
    assert first["source_replay_sha256"] == hashlib.sha256(canonical).hexdigest()
    annotated = json.loads(build_client_package(doc).replay)
    again = json.loads(build_client_package(annotated).assurance)
    assert again["source_replay_sha256"] == first["source_replay_sha256"]


@pytest.mark.parametrize(
    ("change", "outcome"),
    [("unmet", "objectives_unmet"), ("recovered", "incomplete")],
)
def test_a_complete_phase_is_not_completed_when_f1_says_otherwise(change, outcome):
    """The status never contradicts the embedded F1 outcome."""
    doc = replay("complete", completed=0 if change == "unmet" else 1)
    if change == "recovered":
        doc["meta"]["recovery"] = {"recovered_from_tick": 40}
    report = json.loads(build_client_package(doc).assurance)
    assert report["outcome"]["outcome"] == outcome
    assert report["status"] == "incomplete"


def test_the_cli_reads_the_replay_through_the_size_cap(tmp_path, monkeypatch):
    """The replay-input cap applies before parsing: the same valid replay
    packages under the normal cap and is refused under a tiny one."""
    import argusnet.core.io_limits as io_limits

    source = Path(__file__).parent / "fixtures" / "replay_contract_fixture.json"
    main(["client-package", "--replay", str(source), "--output", str(tmp_path / "ok")])
    assert (tmp_path / "ok" / "manifest.json").exists()
    capped = io_limits.read_text_capped
    monkeypatch.setattr(io_limits, "read_text_capped", lambda path, **_: capped(path, max_bytes=16))
    with pytest.raises(SystemExit, match="byte limit"):
        main(["client-package", "--replay", str(source), "--output", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
