"""S1 contracts (docs/contracts): the schemas, and the records a package carries.

Every record a client package carries validates against its kind's schema,
carries the identity and revision keys (VIEWER_UI 10.2) with a sequence
unique across the package, and points into the replay it came from. A kind
no producer records says why in the index, rather than reading as empty.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft7Validator
from referencing import Registry, Resource

from argusnet.evaluation.assurance import build_client_package
from argusnet.evaluation.records import KINDS, NOT_RECORDED
from tests.test_client_package import replay

CONTRACTS = Path(__file__).resolve().parent.parent / "docs" / "contracts"
IDS = ("session_id", "run_id", "epoch", "tick", "world_revision", "belief_revision", "sequence")


def schemas() -> dict[str, dict]:
    return {
        path.name.removesuffix(".schema.json"): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(CONTRACTS.glob("*.schema.json"))
    }


def validator(name: str) -> Draft7Validator:
    loaded = schemas()
    registry: Registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in loaded.values()
    )
    return Draft7Validator(loaded[name], registry=registry)


def package_records(doc: dict) -> tuple[dict, dict[str, list[dict]]]:
    package = build_client_package(doc)
    files = dict(package.records)
    index = json.loads(files["records/index.json"])
    by_kind = {
        kind: json.loads(files[f"records/{kind}.json"])
        for kind, entry in index["kinds"].items()
        if entry["status"] == "recorded"
    }
    return index, by_kind


def check(doc: dict) -> tuple[dict, dict[str, list[dict]]]:
    """Validate everything a package of `doc` carries; return it."""
    index, by_kind = package_records(doc)
    errors = sorted(validator("record-index").iter_errors(index), key=str)
    assert not errors, errors[0].message
    sequences = []
    for kind, records in by_kind.items():
        schema = validator(kind)
        for record in records:
            errors = sorted(schema.iter_errors(record), key=str)
            assert not errors, f"{kind} {record['record_id']}: {errors[0].message}"
            assert set(record["ids"]) == set(IDS)
            sequences.append(record["ids"]["sequence"])
            source = record["source"]
            if isinstance(source, str) and source.startswith("/frames/") and "*" not in source:
                target: object = doc
                for part in source.strip("/").split("/"):
                    target = target[int(part)] if isinstance(target, list) else target[part]  # type: ignore[index]
    assert sorted(sequences) == list(range(len(sequences))), "sequences unique, from 0"
    return index, by_kind


def test_every_schema_is_valid_and_every_kind_has_one() -> None:
    loaded = schemas()
    for name, schema in loaded.items():
        Draft7Validator.check_schema(schema)
        assert schema["$id"].endswith(f"/{name}.schema.json")
    assert set(KINDS) <= set(loaded)
    for kind in KINDS:
        contract = loaded[kind]["allOf"][1]["properties"]["contract"]["const"]
        assert contract == f"argusnet.{kind}/v1"


def test_a_package_carries_every_kind_or_says_why_not() -> None:
    index, by_kind = check(replay())
    assert set(index["kinds"]) == set(KINDS)
    for kind in KINDS:
        entry = index["kinds"][kind]
        if kind in NOT_RECORDED:
            assert entry == {"status": "unavailable", "count": 0, "reason": NOT_RECORDED[kind]}
            assert kind not in by_kind
        else:
            assert entry["status"] == "recorded"
            assert entry["count"] == len(by_kind[kind])
    assert [r["objective_id"] for r in by_kind["mission"][0]["objectives"]] == ["asset"]
    assert by_kind["platform"][0]["platform_id"] == "drone"
    assert by_kind["platform"][0]["firmware_version"]["status"] == "unavailable"
    assert by_kind["localization"][0]["integrity"]["status"] == "unavailable"
    assert by_kind["evidence"][0]["sufficiency"]["status"] == "unavailable"
    assert by_kind["report"][0]["status"] == "completed"


def test_findings_are_the_f1_reasons_and_an_abort_is_a_safety_record() -> None:
    _, by_kind = check(replay("egress", 0, abort=True))
    codes = [finding["code"] for finding in by_kind["finding"]]
    assert "safety_abort" in codes
    abort = next(r for r in by_kind["safety"] if r["validation_id"] == "val-9")
    assert abort["accepted"] is False and abort["reason"] == "abort"
    assert abort["violations"] == ["min_terrain_clearance"]
    assert abort["ids"]["tick"] == 0
    assert by_kind["report"][0]["status"] == "aborted"
    assert by_kind["mission"][0]["outcome"] == "aborted"


def test_missing_identity_is_stated_not_invented() -> None:
    _, by_kind = check(replay())
    ids = by_kind["site"][0]["ids"]
    for key in ("session_id", "run_id", "epoch", "world_revision"):
        assert ids[key]["status"] == "unavailable" and ids[key]["reason"]
    # A run-level record has no tick: stated, not tick 0.
    assert ids["tick"] == {"status": "unavailable", "reason": "a run-level record"}
    # A safety event that does not say whether it was accepted is not read
    # as accepted.
    (event,) = [r for r in by_kind["safety"] if r["record_id"] == "safe-1"]
    assert event["accepted"]["status"] == "unavailable"
    doc = replay()
    doc["meta"].update(session_id="s-1", run_id="r-7", epoch=2, world_revision="w-abc")
    _, by_kind = check(doc)
    ids = by_kind["site"][0]["ids"]
    assert (ids["session_id"], ids["run_id"], ids["epoch"], ids["world_revision"]) == (
        "s-1",
        "r-7",
        2,
        "w-abc",
    )


def test_records_are_deterministic() -> None:
    assert build_client_package(replay()).records == build_client_package(replay()).records


def test_a_real_run_validates_with_a_prior_world_and_without(tmp_path: Path) -> None:
    """A real `argusnet-rs` run, once with the true world as its prior and
    once building its own map: every record validates, and the spatial
    revision says which it was."""
    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "format": "argusnet.scenario-config",
                "version": 1,
                "seed": 7,
                "duration_s": 20.0,
                "options": {"map_preset": "small", "terrain_preset": "urban_flat"},
            }
        ),
        encoding="utf-8",
    )
    for knowledge in ("prior", "none"):
        path = tmp_path / f"{knowledge}.json"
        command = [str(binary), "sim", "--quiet", "--config", str(config)]
        if knowledge == "none":
            command += ["--world-knowledge", "none"]
        subprocess.run([*command, "--replay", str(path)], check=True, capture_output=True)
        doc = json.loads(path.read_text(encoding="utf-8"))
        index, by_kind = check(doc)
        revision = by_kind["spatial_revision"][0]
        assert revision["world_knowledge"] == knowledge
        if knowledge == "none":
            assert (
                isinstance(revision["belief_revisions"], int) and revision["belief_revisions"] > 0
            )
            assert revision["known_fraction"] > 0.0
        else:
            assert revision["belief_revisions"]["status"] == "unavailable"
        assert index["kinds"]["trajectory"]["count"] >= 1
        assert index["kinds"]["platform"]["count"] >= index["kinds"]["trajectory"]["count"]
