#!/usr/bin/env python3
"""Verify every proto field has a recorded disposition in the G0 field matrix.

The G0 guardrail in docs/migration/DELIVERY_PLAN.md asks that every payload be
"preserved, normalized explicitly or rejected; no silent missing-field success".
The failure this prevents is not a field that is dropped -- several are dropped
on purpose -- but a field that is dropped *without anyone having said so*.

So this does not try to decide whether a field is handled correctly. It asserts
only that someone wrote down what is supposed to happen to it. Adding a field to
the proto without a matrix row fails the check; so does leaving a row behind for
a field that no longer exists.

Usage:
    python3 scripts/check_field_parity.py            # verify (exit 1 on drift)
    python3 scripts/check_field_parity.py --list     # print fields, no matrix
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO = REPO_ROOT / "proto" / "argusnet" / "v1" / "world_model.proto"
MATRIX = REPO_ROOT / "docs" / "migration" / "FIELD_PARITY.md"

# Dispositions a matrix row may declare. Anything else is a typo, and a typo
# that silently passed would defeat the point of the check.
DISPOSITIONS = {
    # Reaches every consumer that asks for it.
    "carried",
    # Deliberately not represented in the Rust runtime type. The row must say
    # where it does survive (usually the Python adapter and the replay JSON).
    "dropped-intentional",
    # On the wire and in the generated bindings, but no implementation reads or
    # writes a meaningful value on either side.
    "inert",
    # Preserved only because one code path copies it around the runtime type,
    # so a different RPC answering about the same frame may omit it.
    "grafted",
    # Produced by one side, never consumed by the other. No data is lost on the
    # wire; the far side simply ignores it.
    "consumer-gap",
    # Computed from another field when encoding, and never read back. Carries no
    # information of its own, so it cannot round-trip.
    "derived",
}


def proto_fields() -> dict[str, list[str]]:
    """Map message name -> field names, in declaration order."""
    source = re.sub(r"//.*", "", PROTO.read_text())
    messages: dict[str, list[str]] = {}
    for match in re.finditer(r"\bmessage\s+(\w+)\s*\{", source):
        name = match.group(1)
        index, depth = match.end(), 1
        while depth and index < len(source):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
            index += 1
        body = source[match.end() : index - 1]
        messages[name] = [
            field
            for field, _ in re.findall(
                r"^\s*(?:repeated\s+|optional\s+)?[\w.<>, ]+?\s+(\w+)\s*=\s*(\d+)\s*;",
                body,
                re.M,
            )
        ]
    return messages


def matrix_rows() -> dict[str, str]:
    """Map "Message.field" -> disposition, from the markdown table."""
    if not MATRIX.exists():
        sys.exit(f"missing field matrix: {MATRIX.relative_to(REPO_ROOT)}")
    rows: dict[str, str] = {}
    # Only the table under "## Matrix". The dispositions legend earlier in the
    # file is also a markdown table of backticked cells, and parsing it would
    # silently invent rows named after the vocabulary itself.
    body = MATRIX.read_text().split("## Matrix", 1)
    if len(body) != 2:
        sys.exit(f"{MATRIX.relative_to(REPO_ROOT)} has no '## Matrix' section")
    for line in body[1].splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        key = cells[0].strip("`")
        disposition = cells[1].strip("`")
        rows[key] = disposition
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every proto field and exit, without consulting the matrix.",
    )
    args = parser.parse_args()

    fields = proto_fields()
    declared = {f"{message}.{field}" for message, names in fields.items() for field in names}

    if args.list:
        for key in sorted(declared):
            print(key)
        return 0

    recorded = matrix_rows()
    problems: list[str] = []

    undocumented = sorted(declared - set(recorded))
    if undocumented:
        problems.append(
            "proto fields with no row in the field matrix "
            f"({len(undocumented)}):\n  " + "\n  ".join(undocumented)
        )

    stale = sorted(set(recorded) - declared)
    if stale:
        problems.append(
            f"matrix rows for fields that no longer exist in the proto ({len(stale)}):\n  "
            + "\n  ".join(stale)
        )

    bad = sorted(
        f"{key} -> {value}" for key, value in recorded.items() if value not in DISPOSITIONS
    )
    if bad:
        problems.append(
            f"rows with an unrecognized disposition ({len(bad)}); "
            f"valid values are {sorted(DISPOSITIONS)}:\n  " + "\n  ".join(bad)
        )

    if problems:
        print("\n\n".join(problems), file=sys.stderr)
        print(
            f"\nUpdate {MATRIX.relative_to(REPO_ROOT)} so every field has a stated "
            "disposition.\nList the current fields with:\n"
            "    python3 scripts/check_field_parity.py --list",
            file=sys.stderr,
        )
        return 1

    print(f"Field matrix covers all {len(declared)} proto fields across {len(fields)} messages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
