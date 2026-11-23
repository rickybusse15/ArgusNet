"""Analysis imports without the native library (P6.1).

Evaluation, reporting, offline import and configuration are usable from a
checkout or an install where Rust was never built: the native library is loaded
when something calls into it, not when a module is imported. Every module is
imported first, on its own, so no import order hides a circle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import importlib, sys

class NoNative:
    def find_spec(self, name, path=None, target=None):
        if name == "argusnet._native":
            raise ImportError("the native library is not built")
        return None

sys.meta_path.insert(0, NoNative())
failed = []
for name in sys.argv[1:]:
    for loaded in [m for m in sys.modules if m == "argusnet" or m.startswith("argusnet.")]:
        del sys.modules[loaded]
    try:
        importlib.import_module(name)
    except BaseException as error:
        failed.append(f"{name}: {type(error).__name__}: {error}")
print("\\n".join(failed))
"""


@pytest.mark.parametrize(
    "package", ["cli", "core", "evaluation", "mapping", "mission", "planning", "security", "world"]
)
def test_every_module_imports_first_and_without_the_native_library(package):
    source = ROOT / "src"
    modules = sorted(
        ".".join(path.relative_to(source).with_suffix("").parts).removesuffix(".__init__")
        for path in (source / "argusnet" / package).rglob("*.py")
        if path.name != "__main__.py"
    )
    assert modules
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, *modules],
        cwd=ROOT,
        env={"PYTHONPATH": str(source), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
