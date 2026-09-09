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

    # Prove strictness behaviourally rather than trusting the import: a document
    # the schema rejects must actually be rejected.
    valid_frame = {
        "timestamp_s": 0.0,
        "nodes": [],
        "observations": [],
        "rejected_observations": [],
        "tracks": [],
        "truths": [],
        "metrics": {},
    }
    meta = {"schema_version": "1.0.0", "seed": 1, "dt_s": 0.1, "duration_s": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        wrong_type = replay.validate_replay_with_schema(
            {
                "meta": meta,
                "frames": [{**valid_frame, "tracks": [{"track_id": 1234, "position": "nope"}]}],
            }
        )
        unknown_key = replay.validate_replay_with_schema(
            {"meta": meta, "frames": [valid_frame], "an_unexpected_key": 1}
        )
    check(
        "a wrong-typed nested track field is rejected",
        len(wrong_type) > 0,
        "accepted a track whose id is an int and position a string",
    )
    check(
        "an unknown top-level key is rejected",
        len(unknown_key) > 0,
        "accepted a document with a key the schema does not allow",
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
