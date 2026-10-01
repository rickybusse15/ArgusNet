"""The native builder's constant tables are the reference's (P4.9b.2b).

`rust/argusnet-sim/src/builder/reference_tables.json` is generated from the
Python reference by `scripts/export_builder_tables.py` and embedded in the
Rust builder. A change to a preset, a constant or a default on the Python side
must be re-exported, or the two builders silently diverge.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_the_embedded_tables_match_the_reference():
    spec = importlib.util.spec_from_file_location(
        "export_builder_tables", ROOT / "scripts" / "export_builder_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = (
        ROOT / "rust" / "argusnet-sim" / "src" / "builder" / "reference_tables.json"
    ).read_text()
    assert committed == module.render(), "stale: rerun scripts/export_builder_tables.py"
