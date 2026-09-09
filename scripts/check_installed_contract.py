#!/usr/bin/env python3
"""Assert the installed package honours its contracts, outside the source tree.

G0 asks for an installed-wheel contract job. The failure it guards against is a
package that works from a checkout and quietly stops working once installed --
a data file that was never packaged, a path computed by walking up from
``__file__``, or a dependency that only the dev extra pulls in.

Run this with the interpreter of a clean environment that has the wheel
installed, from a working directory that is **not** the repository:

    python -m venv /tmp/contract-venv
    /tmp/contract-venv/bin/pip install dist/argusnet-*.whl
    cd /tmp && /tmp/contract-venv/bin/python .../check_installed_contract.py

Running it from the repository root would let the source tree satisfy imports
and data-file lookups that the wheel does not, which is the whole thing being
tested, so it refuses to run there.
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

FAILURES: list[str] = []


def check(description: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {description}")
    else:
        print(f"  FAIL  {description}{': ' + detail if detail else ''}")
        FAILURES.append(description)


def _minimal_valid_document() -> dict:
    """A replay document the packaged schema accepts with zero errors.

    Kept in step with `tests/_minimal_valid_document`, which cannot be imported
    here: the test suite is not part of the wheel, and this script has to run
    against an installation with no checkout present.
    """
    return {
        "meta": {
            "scenario_name": "unit-test",
            "generated_at_utc": "2025-01-15T12:00:00+00:00",
            "frame_count": 1,
            "dt_s": 0.5,
            "seed": 42,
            "node_ids": ["node-a"],
            "track_ids": ["track-1"],
        },
        "summary": {
            "mean_error_m": 1.5,
            "peak_error_m": 3.0,
            "mean_observations_per_frame": 2.0,
            "mean_active_tracks": 1.0,
            "mean_accepted_observations_per_frame": 2.0,
            "mean_rejected_observations_per_frame": 0.0,
            "total_accepted_observations": 2,
            "total_rejected_observations": 0,
            "observation_rejection_rate": 0.0,
            "rejection_counts": {},
        },
        "frames": [
            {
                "timestamp_s": 0.0,
                "nodes": [
                    {
                        "node_id": "node-a",
                        "position": [0.0, 0.0, 0.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "is_mobile": False,
                        "timestamp_s": 0.0,
                        "health": 1.0,
                        "sensor_type": "optical",
                        "fov_half_angle_deg": 180.0,
                        "max_range_m": 0.0,
                    }
                ],
                "observations": [
                    {
                        "node_id": "node-a",
                        "target_id": "target-1",
                        "origin": [0.0, 0.0, 0.0],
                        "direction": [1.0, 0.0, 0.0],
                        "bearing_std_rad": 0.002,
                        "timestamp_s": 0.0,
                        "confidence": 1.0,
                    }
                ],
                "rejected_observations": [],
                "tracks": [
                    {
                        "track_id": "track-1",
                        "timestamp_s": 0.0,
                        "position": [50.0, 15.0, 10.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "covariance": [
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            1.0,
                        ],
                        "measurement_std_m": 5.0,
                        "update_count": 1,
                        "stale_steps": 0,
                    }
                ],
                "truths": [
                    {
                        "target_id": "target-1",
                        "position": [50.0, 15.0, 10.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "timestamp_s": 0.0,
                    }
                ],
                "metrics": {
                    "mean_error_m": 1.5,
                    "max_error_m": 1.5,
                    "active_track_count": 1,
                    "observation_count": 2,
                    "accepted_observation_count": 2,
                    "rejected_observation_count": 0,
                    "mean_measurement_std_m": 5.0,
                    "track_errors_m": {},
                    "rejection_counts": {},
                    "accepted_observations_by_target": {},
                    "rejected_observations_by_target": {},
                },
                "generation_rejections": [],
                "mapping_state": None,
                "localization_state": None,
                "inspection_events": [],
            }
        ],
    }


def refuse_to_run_from_the_source_tree() -> None:
    """A checkout in the working directory would mask exactly what this tests."""
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "argusnet").is_dir():
            sys.exit(
                f"refusing to run from inside the source tree ({candidate}).\n"
                "The checkout would satisfy imports and data files that the "
                "installed wheel must provide on its own. Run this from a "
                "directory outside the repository."
            )


def main() -> int:
    refuse_to_run_from_the_source_tree()

    import argusnet

    package_dir = Path(argusnet.__file__).resolve().parent
    print(f"installed package: {package_dir}")
    check(
        "package is imported from an installation, not the source tree",
        "site-packages" in str(package_dir) or "dist-packages" in str(package_dir),
        str(package_dir),
    )

    # --- the replay schema ships, and is the real one -----------------------
    from argusnet.evaluation import replay

    schema = replay._load_replay_schema()
    check(
        "replay schema is packaged and loadable",
        isinstance(schema, dict) and "properties" in schema,
    )
    check(
        "replay schema is the full document, not a stub",
        "frames" in schema.get("properties", {}) and "meta" in schema.get("properties", {}),
        f"top-level properties: {sorted(schema.get('properties', {}))}",
    )

    # --- and the validator that enforces it ships too ------------------------
    # Shipping the schema without jsonschema silently downgrades validation to a
    # hand-written check that misses unknown keys, wrong-typed nested fields and
    # out-of-range values. The schema being present is not enough on its own.
    try:
        import jsonschema  # noqa: F401

        has_jsonschema = True
    except ImportError:
        has_jsonschema = False
    check("jsonschema is installed as a runtime dependency", has_jsonschema)

    # Prove strictness behaviourally rather than trusting the import.
    #
    # This has to start from a document the schema accepts. An earlier version
    # did not: its `metrics` was `{}`, which `platform_metrics` rejects for
    # four missing required fields, so every mutation "failed validation" for a
    # reason unrelated to the field under test. The base document is asserted
    # to produce zero errors first, so if it ever drifts from the schema this
    # fails loudly instead of quietly making the probes meaningless.
    base = _minimal_valid_document()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        baseline = replay.validate_replay_with_schema(base)
    check(
        "the probe's own base document validates cleanly",
        not baseline,
        f"base document is not schema-valid, so the mutations below prove nothing: {baseline}",
    )

    def mutated(apply):
        document = copy.deepcopy(base)
        apply(document)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            return replay.validate_replay_with_schema(document)

    # Each mutation changes exactly one thing the schema forbids. Note what is
    # deliberately absent: an unknown top-level key. The schema sets
    # `additionalProperties: true` and tests/test_replay_schema.py asserts extra
    # fields are accepted for forward compatibility, so treating one as invalid
    # would assert the opposite of the contract.
    wrong_typed_track_id = mutated(
        lambda d: d["frames"][0]["tracks"][0].__setitem__("track_id", 1234)
    )
    check(
        "a wrong-typed nested track field is rejected",
        len(wrong_typed_track_id) > 0,
        "accepted a track whose id is an integer",
    )

    wrong_typed_node_position = mutated(
        lambda d: d["frames"][0]["nodes"][0].__setitem__("position", [1.0, 2.0, "three"])
    )
    check(
        "a wrong-typed nested node field is rejected",
        len(wrong_typed_node_position) > 0,
        "accepted a node position containing a string",
    )

    missing_required_metrics = mutated(lambda d: d["frames"][0].__setitem__("metrics", {}))
    check(
        "a frame missing required metrics is rejected",
        len(missing_required_metrics) > 0,
        "accepted a frame whose metrics omit every required field",
    )

    # --- the native world library ships --------------------------------------
    from argusnet import _native_world

    try:
        _native_world._library()
        native_ok, native_detail = True, ""
    except Exception as error:  # noqa: BLE001 - reporting, not handling
        native_ok, native_detail = False, str(error)
    check("native world library loads from the installation", native_ok, native_detail)

    # --- console entry points resolve ---------------------------------------
    from importlib.metadata import entry_points

    scripts = {ep.name for ep in entry_points(group="console_scripts")}
    for name in ("argusnet", "argusnet-sim", "argusnet-export", "argusnet-build-scene"):
        check(f"console script '{name}' is registered", name in scripts)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} contract check(s) failed:", file=sys.stderr)
        for failure in FAILURES:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Installed package honours its contracts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
